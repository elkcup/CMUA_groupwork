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
import bim_attack as bim_attack

from StandardPGD.PGD import PGDAttack
from MIM.mim import MIMattack

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

