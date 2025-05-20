import argparse
import json
import os
from os.path import join
from model_data_prepare import prepare
import matplotlib.image
from tqdm import tqdm
import copy

import torch
import torch.utils.data as data
import torchvision.utils as vutils
import torch.nn.functional as F

import sys
from contextlib import contextmanager
import logging


from AttGAN.data import check_attribute_conflict
from data import CelebA


import attacks
import auto_pgd_attacks as apgd_attacks
from auto_fgsm_attacks import FGSMAttack
from StandardPGD.PGD import PGDAttack
from MIM.mim import MIMattack
import bim_attack as bim_attack


from evaluate_pgd_mim import evaluate_pgd, evaluate_mim
from evaluate_cmua_apgd import evaluate_apgd,evaluate_cmua
from auto_fgsm_evaluate import evaluate_FGSM
from bim_evaluate import evaluate_bim

import matplotlib.pyplot as plt
import numpy as np

# 屏蔽部分输出
@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull

        logging.basicConfig(level=logging.ERROR)
        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        old_levels = {}
        for logger in loggers:
            old_levels[logger.name] = logger.level
            logger.setLevel(logging.ERROR)
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            for logger in loggers:
                if logger.name in old_levels:
                    logger.setLevel(old_levels[logger.name])


# 加载配置
def parse_config(path):
    """解析配置文件为参数对象"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, path)
    with open(config_path, 'r') as f:
        args_attack = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
    return args_attack


# 初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.system('cls' if os.name == 'nt' else 'clear')

# 加载模型和数据
print("加载模型和数据...")
with suppress_output():
    attack_dataloader, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = prepare()
print("模型和数据加载完成！")
print('-------------------------------------------------------')


print('开始训练')
# 定义攻击函数 针对单张图片
def attacking_models(args_attack,attack_dataloader, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models,pgd_attack,save_path):
    for idx, (img_a, att_a, c_org) in enumerate(attack_dataloader):
        img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
        att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
        att_a = att_a.type(torch.float)

        if idx == 0:
            img_a_last = copy.deepcopy(img_a)

        # # attack stargan
        solver.test_universal_model_level_attack(idx, img_a, c_org, pgd_attack)

        # attack attentiongan
        attentiongan_solver.test_universal_model_level_attack(idx, img_a, c_org, pgd_attack)

        # attack HiSD
        with torch.no_grad():
            c = E(img_a)
            c_trg = c
            s_trg = F(reference, 1)
            c_trg = T(c_trg, s_trg, 1)
            x_trg = G(c_trg)
            mask = abs(x_trg - img_a)
            mask = mask[0,0,:,:] + mask[0,1,:,:] + mask[0,2,:,:]
            mask[mask>0.5] = 1
            mask[mask<0.5] = 0
        pgd_attack.universal_perturb_HiSD(img_a.cuda(), transform, F, T, G, E, reference, x_trg+0.002, gen_models, mask)

        # attack AttGAN
        att_b_list = [att_a]
        for i in range(attgan_args.n_attrs):
            tmp = att_a.clone()
            tmp[:, i] = 1 - tmp[:, i]
            tmp = check_attribute_conflict(tmp, attgan_args.attrs[i], attgan_args.attrs)
            att_b_list.append(tmp)

        for i, att_b in enumerate(att_b_list):
            att_b_ = (att_b * 2 - 1) * attgan_args.thres_int
            if i > 0:
                att_b_[..., i - 1] = att_b_[..., i - 1] * attgan_args.test_int / attgan_args.thres_int
            with torch.no_grad():
                gen_noattack = attgan.G(img_a, att_b_)
            x_adv, perturb = pgd_attack.universal_perturb_attgan(img_a, att_b_, gen_noattack, attgan)

        torch.save(pgd_attack.up, save_path)
        break
        


def attacking_pgd_mim(args_attack,attack_dataloader, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models,model,save_path,best_params):
    perturb_stargan_list=[]
    perturb_attention_list=[]
    perturb_hisd_list=[]
    perturb_attgan_list=[]

    for idx, (img_a, att_a, c_org) in enumerate(attack_dataloader):
        img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
        att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
        att_a = att_a.type(torch.float)

        if idx == 0:
            img_a_last = copy.deepcopy(img_a)

        batch_star=[]

        solver.G.to(device)
        c_trg_list = solver.create_labels(c_org, solver.c_dim, solver.dataset, solver.selected_attrs)
        for c_trg in c_trg_list:
            with torch.no_grad():
                out_put, _ = solver.G(img_a, c_trg)
        
            _,perturb_stargan=model.perturb_stargan(img_a,out_put,c_trg,solver.G)
            batch_star.append(perturb_stargan)
        perturb_stargan = torch.stack(batch_star, dim=0).mean(dim=0)
        perturb_stargan_list.append(perturb_stargan)

        # 2. 攻击AttentionGAN
        batch_attention = []
        attentiongan_solver.G.to(device)

        c_trg_list = attentiongan_solver.create_labels(
            c_org, attentiongan_solver.c_dim, attentiongan_solver.dataset, attentiongan_solver.selected_attrs
        )

        for c_trg in c_trg_list:
            with torch.no_grad():
                output, _, _ = attentiongan_solver.G(img_a, c_trg)
            _, perturb_attention = model.perturb_attentiongan(img_a, output, c_trg, attentiongan_solver.G)
            batch_attention.append(perturb_attention)
        perturb_attention = torch.stack(batch_attention, dim=0).mean(dim=0)
        perturb_attention_list.append(perturb_attention)

        # 3. 攻击HiSD
        with torch.no_grad():
            c = E(img_a)
            c_trg = c
            s_trg = F(reference, 1)
            c_trg = T(c_trg, s_trg, 1)
            x_trg = G(c_trg)
            mask = abs(x_trg - img_a).sum(dim=1, keepdim=True)
            mask = (mask > 0.5).float()
        
        _, perturb_hisd = model.perturb_hisd(
            img_a, reference, x_trg, E, F, T, G, gen_models, mask, transform
        )
        perturb_hisd_list.append(perturb_hisd)

        # 4. 攻击AttGAN
        batch_attgan = []
        attgan.G.to(device)
        att_b_list = [att_a]
        
        for i in range(attgan_args.n_attrs):
            tmp= att_a.clone()
            tmp[:, i] = 1 - tmp[:, i]
            tmp = check_attribute_conflict(tmp, attgan_args.attrs[i], attgan_args.attrs)
            att_b_list.append(tmp)
        
        for i, att_b in enumerate(att_b_list):
            att_b_ = (att_b * 2 - 1) * attgan_args.thres_int
            if i > 0:
                att_b_[..., i - 1] = att_b_[..., i - 1] * attgan_args.test_int / attgan_args.thres_int

            with torch.no_grad():
                gen_noattack = attgan.G(img_a, att_b_)
            
            _, perturb_attgan =model.perturb_attgan(img_a, att_b_, gen_noattack, attgan)
            batch_attgan.append(perturb_attgan)
        perturb_attgan = torch.stack(batch_attgan, dim=0).mean(dim=0)
        perturb_attgan_list.append(perturb_attgan)

        # 保存各模型的扰动
        name=None
        if model.__class__.__name__ == 'PGDAttack':
            name = 'pgd'
        else:
            name = 'mim'
        perturb_stargan = torch.clamp(torch.stack(perturb_stargan_list, dim=0).mean(dim=0), min=-best_params.pgd_stargan.epsilon, max=+best_params.pgd_stargan.epsilon)
        perturb_attention = torch.clamp(torch.stack(perturb_attention_list, dim=0).mean(dim=0), min=-best_params.pgd_attentiongan.epsilon, max=+best_params.pgd_attentiongan.epsilon)
        perturb_hisd = torch.clamp(torch.stack(perturb_hisd_list, dim=0).mean(dim=0), min=-best_params.pgd_hisd.epsilon, max=+best_params.pgd_hisd.epsilon)
        perturb_attgan = torch.clamp(torch.stack(perturb_attgan_list, dim=0).mean(dim=0), min=-best_params.pgd_attgan.epsilon, max=+best_params.pgd_attgan.epsilon)
        
        os.makedirs(save_path, exist_ok=True)
        torch.save(perturb_stargan, os.path.join(save_path, f'{name}_stargan_perturbation_bad.pt'))
        torch.save(perturb_attention, os.path.join(save_path, f'{name}_attentiongan_perturbation_bad.pt'))
        torch.save(perturb_hisd, os.path.join(save_path, f'{name}_hisd_perturbation_bad.pt'))
        torch.save(perturb_attgan, os.path.join(save_path, f'{name}_attgan_perturbation_bad.pt'))
        break
        

#################################
#       CMAU和AutoPGD攻击       # 生成perturbation_bad.pt和auto_pgd_perturbation_bad.pt文件     @Liu Yifei
#################################

# CMAU攻击

# # init the attacker
args_attack = parse_config('setting.json')

pgd_attack = attacks.LinfPGDAttack(model=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), epsilon=args_attack.attacks.epsilon, k=args_attack.attacks.k, a=args_attack.attacks.a, star_factor=args_attack.attacks.star_factor, attention_factor=args_attack.attacks.attention_factor, att_factor=args_attack.attacks.att_factor, HiSD_factor=args_attack.attacks.HiSD_factor, args=args_attack.attacks)

attacking_models(args_attack,attack_dataloader, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models,pgd_attack,'perturbation_bad.pt')
print('save the CMUA-Watermark as perturbation_bad.pt')

# AutoPGD攻击
args_attack = parse_config('auto_pgd_setting.json')
pgd_attack = apgd_attacks.LinfPGDAttack(model=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), epsilon=args_attack.attacks.epsilon, k=args_attack.attacks.k, a=args_attack.attacks.a, star_factor=args_attack.attacks.star_factor, attention_factor=args_attack.attacks.attention_factor, att_factor=args_attack.attacks.att_factor, HiSD_factor=args_attack.attacks.HiSD_factor, args=args_attack.attacks)

attacking_models(args_attack,attack_dataloader, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models,pgd_attack,'auto_pgd_perturbation_bad.pt')
print('save the Auto_PGD_CMUA-Watermark as auto_pgd_perturbation_bad.pt')


#################################
#       DI2-FGSM和M-DI2-FGSM攻击       # 生成DI2_FGSM_perturbation.pt和M_DI2_FGSM_perturbation.pt文件     @Huang Yixuan
#################################

#加载配置文件
args_attack = parse_config('setting.json')

#初始化DI2_FGSM攻击器
DI2_fgsm_attack = FGSMAttack(model=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), momentum = 0.9, args=args_attack.attacks, mode='universal', method='DI2_FGSM')

#基于DI2_FGSM训练扰动
attacking_models(args_attack,attack_dataloader, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models,DI2_fgsm_attack,'DI2_FGSM_perturbation_bad.pt')
print('save the DI2-FGSM-Watermark as DI2_FGSM_perturbation_bad.pt')

#初始化M_DI2_FGSM攻击器
M_DI2_fgsm_attack = FGSMAttack(model=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), momentum = 0.9, args=args_attack.attacks, mode='universal', method='M_DI2_FGSM')

#基于M_DI2_FGSM训练扰动
attacking_models(args_attack,attack_dataloader, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models,M_DI2_fgsm_attack,'M_DI2_FGSM_perturbation_bad.pt')
print('save the M-DI2-FGSM-Watermark as M_DI2_FGSM_perturbation_bad.pt')

#################################
#       PGD和MIM攻击            #      @Chao Wang
#################################

best_params = parse_config('best_params_pgd_mim.json')
#初始化PGD攻击器
pgd=PGDAttack(
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        epsilon=args_attack.pgd_attacks.epsilon,
        iterations=args_attack.pgd_attacks.k,
        step_size=args_attack.pgd_attacks.a,
)

attacking_pgd_mim(args_attack,attack_dataloader, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models,pgd,'./PGD_perturbation_bad',best_params)
print('save the PGD-Watermark as folder PGD_perturbation_bad')

# 初始化MIM攻击器
mim=MIMattack(
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        epsilon=args_attack.mim_attacks.epsilon,
        iterations=args_attack.mim_attacks.iterations,
        decay_factor=args_attack.mim_attacks.decay_factor
)

attacking_pgd_mim(args_attack,attack_dataloader, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models,mim,'./MIM_perturbation_bad',best_params)
print('save the MIM-Watermark as folder MIM_perturbation_bad')


#################################
#       BIM攻击                 # 生成bim_perturbation_bad.pt文件     @Qi Xueting
#################################

# 加载配置文件
args_attack = parse_config('setting.json')

bim_attack = bim_attack.BIMAttack(model=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), epsilon=args_attack.attacks.epsilon, k=args_attack.attacks.k, a=args_attack.attacks.a, star_factor=args_attack.attacks.star_factor, attention_factor=args_attack.attacks.attention_factor, att_factor=args_attack.attacks.att_factor, HiSD_factor=args_attack.attacks.HiSD_factor, args=args_attack.attacks)

attacking_models(args_attack,attack_dataloader, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models,bim_attack,'bim_perturbation_bad.pt')
print('save the BIM-Watermark as bim_perturbation_bad.pt')



print('完成训练过程')

print('------------------------------------------------------------------')
print('现进行评估')
############################################
##               evaluate                ###
############################################

subset = data.Subset(test_dataloader.dataset, list(range(129,134)))
sub_loader = data.DataLoader(subset, batch_size=1, shuffle=False)

#################################
#           PGD和MIM评估        # @Wang Chao
#################################
# 评估PGD攻击
pgd_attack = PGDAttack()
pgd_hisd_dist, pgd_attgan_dist, pgd_attentiongan_dist, pgd_stargan_dist = evaluate_pgd(
    args_attack, sub_loader, attgan, attgan_args, solver, 
    attentiongan_solver, transform, F, T, G, E, reference)
pgd_results = {
    "attgan": pgd_attgan_dist,
    "stargan": pgd_stargan_dist,
    "attentiongan": pgd_attentiongan_dist,
    "hisd": pgd_hisd_dist
}

# 评估MIM攻击
mim_attack = MIMattack()
mim_hisd_dist, mim_attgan_dist, mim_attentiongan_dist, mim_stargan_dist = evaluate_mim(
    args_attack, sub_loader, attgan, attgan_args, solver, 
    attentiongan_solver, transform, F, T, G, E, reference)
mim_results = {
    "attgan": mim_attgan_dist,
    "stargan": mim_stargan_dist,
   "attentiongan": mim_attentiongan_dist,
    "hisd": mim_hisd_dist
}



##################################
##       CMAU和AutoPGD评估       # 需要有auto_pgd_perturbation.pt文件 @Liu Yifei
##################################
# 评估AutoPGD

apgd_hisd_dist,apgd_attgan_dist,apgd_attentiongan_dist, apgd_stargan_dist = evaluate_apgd(args_attack, sub_loader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models)

# 评估CMAU
cmua_hisd_dist,cmua_attgan_dist,cmua_attentiongan_dist, cmua_stargan_dist = evaluate_cmua(args_attack, sub_loader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models)




#################################
#       DI2-FGSM和M-DI2-FGSM评估       # 当前目录需要有DI2_FGSM_perturbation.pt、M_DI2_FGSM_perturbation.pt文件          @Huang Yixuan
#################################

DI2_up=torch.load('DI2_FGSM_perturbation.pt')
M_DI2_up=torch.load('M_DI2_FGSM_perturbation.pt')
#评估DI2-FGSM攻击
D_fgsm_HiDF_sr, D_fgsm_stargan_sr, D_fgsm_AttGAN_sr, D_fgsm_AttentionGAN_sr=evaluate_FGSM(args_attack, sub_loader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models, DI2_up, 'DI2_FGSM')
#评估M-DI2-FGSM攻击
MD_fgsm_HiDF_sr, MD_fgsm_stargan_sr, MD_fgsm_AttGAN_sr, MD_fgsm_AttentionGAN_sr=evaluate_FGSM(args_attack, sub_loader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models, M_DI2_up, 'M_DI2_FGSM')


##################################
##       BIM评估                 # 需要有bim_perturbation.pt文件 @Qi Xueting
##################################

# 评估BIM
bim_hisd_dist,bim_attgan_dist,bim_attentiongan_dist, bim_stargan_dist = evaluate_bim(args_attack, sub_loader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models, bim_attack)


## 结果输出
###### FGSM AND HIDF       @Huang Yixuan
D_fgsm_avg = (D_fgsm_HiDF_sr + D_fgsm_stargan_sr + D_fgsm_AttGAN_sr + D_fgsm_AttentionGAN_sr) / 4
MD_avg = (MD_fgsm_HiDF_sr + MD_fgsm_stargan_sr + MD_fgsm_AttGAN_sr + MD_fgsm_AttentionGAN_sr) / 4


# ##### PGD AND MIN      @Wang Chao
# # 计算平均成功率
pgd_avg = (pgd_hisd_dist + pgd_stargan_dist + pgd_attgan_dist + pgd_attentiongan_dist) / 4
mim_avg = (mim_hisd_dist + mim_stargan_dist + mim_attgan_dist + mim_attentiongan_dist) / 4
# print(f"平均成功率\t{pgd_avg:.4f}\t\t{mim_avg:.4f}")


####### CMUA AND APGD       @Liu Yifei
# 计算平均成功率
cmua_avg = (cmua_hisd_dist + cmua_stargan_dist + cmua_attgan_dist + cmua_attentiongan_dist) / 4
apgd_avg = (apgd_hisd_dist + apgd_stargan_dist + apgd_attgan_dist + apgd_attentiongan_dist) / 4


####### BIM       @Qi Xueting
# 计算平均成功率
bim_avg = (bim_hisd_dist + bim_stargan_dist + bim_attgan_dist + bim_attentiongan_dist) / 4


print("\n========成功率============")
print("模型\t\tCMUA成功率\tAutoPGD成功率\tDI2-FGSM成功率\tM-DI2-FGSM成功率\tPGD成功率\tMIM成功率\tBIM成功率")
print(f"HiSD\t\t{cmua_hisd_dist:.4f}\t\t{apgd_hisd_dist:.4f}\t\t{D_fgsm_HiDF_sr:.4f}\t\t{MD_fgsm_HiDF_sr:.4f}\t\t\t{pgd_hisd_dist:.4f}\t\t{mim_hisd_dist:.4f}\t\t{bim_hisd_dist:.4f}")
print(f"StarGAN\t\t{cmua_stargan_dist:.4f}\t\t{apgd_stargan_dist:.4f}\t\t{D_fgsm_stargan_sr:.4f}\t\t{MD_fgsm_stargan_sr:.4f}\t\t\t{pgd_stargan_dist:.4f}\t\t{mim_stargan_dist:.4f}\t\t{bim_stargan_dist:.4f}")
print(f"AttGAN\t\t{cmua_attgan_dist:.4f}\t\t{apgd_attgan_dist:.4f}\t\t{D_fgsm_AttGAN_sr:.4f}\t\t{MD_fgsm_AttGAN_sr:.4f}\t\t\t{pgd_attgan_dist:.4f}\t\t{mim_attgan_dist:.4f}\t\t{bim_attgan_dist:.4f}")
print(f"AttentionGAN\t{cmua_attentiongan_dist:.4f}\t\t{apgd_attentiongan_dist:.4f}\t\t{D_fgsm_AttentionGAN_sr:.4f}\t\t{MD_fgsm_AttentionGAN_sr:.4f}\t\t\t{pgd_attentiongan_dist:.4f}\t\t{mim_attentiongan_dist:.4f}\t\t{bim_attentiongan_dist:.4f}")


print(f"平均成功率\t{cmua_avg:.4f}\t\t{apgd_avg:.4f}\t\t{D_fgsm_avg:.4f}\t\t{MD_avg:.4f}\t\t\t{pgd_avg:.4f}\t\t{mim_avg:.4f}\t\t{bim_avg:.4f}")



#################################
#       绘制图片
#################################


# 模型名称
models = ['AttGAN', 'stargan', 'AttentionGAN', 'HiSD']
num_models = len(models)

# 每个模型对应的7个变量
data = {
    'AttGAN': [
        pgd_attgan_dist,       
        mim_attgan_dist,       
        cmua_attgan_dist,      
        apgd_attgan_dist,      
        D_fgsm_AttGAN_sr,      
        MD_fgsm_AttGAN_sr,
        bim_attgan_dist
    ],
    'stargan': [
        pgd_stargan_dist,
        mim_stargan_dist,
        cmua_stargan_dist,
        apgd_stargan_dist,
        D_fgsm_stargan_sr,
        MD_fgsm_stargan_sr,
        bim_stargan_dist
    ],
    'AttentionGAN': [
        pgd_attentiongan_dist,
        mim_attentiongan_dist,
        cmua_attentiongan_dist,
        apgd_attentiongan_dist,
        D_fgsm_AttentionGAN_sr,
        MD_fgsm_AttentionGAN_sr,
        bim_attentiongan_dist
    ],
    'HiSD': [
        pgd_hisd_dist,
        mim_hisd_dist,
        cmua_hisd_dist,
        apgd_hisd_dist,
        D_fgsm_HiDF_sr,
        MD_fgsm_HiDF_sr,
        bim_hisd_dist
    ]
}

# 将数据转换为4x6的二维数组（行：模型，列：变量）
data_array = np.array([data[model] for model in models])

# 设置参数
spacing = 2.5  # 模型组之间的间隔
width = 0.3     # 每个条形的宽度
num_bars_per_model = 7

# 生成中心位置（使用更大的间隔）
x = np.arange(num_models) * spacing  # 模型组的中心位置

# 创建图形
fig, ax = plt.subplots(figsize=(16, 6))  # 加宽图形

# 定义颜色和标签
colors = plt.cm.tab20.colors[:num_bars_per_model]  # 使用10种颜色的前6种
labels = [
    'PGD', 'MIM', 'CMUA', 'AutoPGD', 'D-FGSM', 'MD-FGSM', 'BIM'
]  # 根据变量名提取攻击方法作为标签

# 绘制每个条形
for i in range(num_bars_per_model):
    # 计算每个条形的x位置（围绕模型中心对称）
    offset = (i - (num_bars_per_model - 1)/2) * width * 1.1 
    ax.bar(
        x + offset,
        data_array[:, i],  # 当前列的数据
        width=width,
        color=colors[i],
        label=labels[i] if i < len(labels) else f'Category {i+1}'
    )

# 设置坐标轴标签和刻度
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel('L2_mask')
ax.set_title('L2_mask Comparison Across Models')

# 添加图例和调整布局
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncols=3)
plt.tight_layout(rect=[0, 0, 0.85, 1])  # 为图例保留空间

# 显示图形
plt.savefig('SR.png')
