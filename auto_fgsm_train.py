import argparse
import copy
import json
import os
from os.path import join
import sys
import matplotlib.image
from tqdm import tqdm

import torch
import torch.utils.data as data
import torchvision.utils as vutils
import torch.nn.functional as F

from AttGAN.data import check_attribute_conflict

from data import CelebA
import auto_fgsm_attacks as attacks

from auto_fgsm_model_data_prepare import prepare
from auto_fgsm_evaluate import evaluate_AttGAN, evaluate_stargan, evaluate_AttentionGAN, evaluate_HIDF, evaluate_FGSM

#字典化，方便调用
class ObjDict(dict):
    def __getattr__(self,name):
        try:
            return self[name]
        except:
            raise AttributeError(name)
    def __setattr__(self,name,value):
        self[name]=value
        
#加载参数
def parse(args=None):
    with open(join('./fgsm_setting.json'), 'r') as f:
        args_attack = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
    return args_attack

#创建攻击器
def init_Attack(args_attack):
    fgsm_attack = attacks.FGSMAttack(model=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), momentum = 0.9, args=args_attack.attacks, mode='universal', method='M_DI2_FGSM')#在这里设置method，表示选择DI2-FGSM或M-DI2-FGSM;设置训练模式（通用 or 专用）,可以取值'universal','attgan','stargan','attention','HiSD'
    return fgsm_attack

########################################

#初始化参数
args_attack = parse()
print(args_attack)

#初始化攻击器
fgsm_attack = init_Attack(args_attack)

#########################################
if fgsm_attack.mode=='universal':
    #创建目录
    os.system('copy -r ./results {}/results{}'.format(args_attack.global_settings.results_path, args_attack.attacks.momentum))
    print("实验目录已创建")
    
    # 保存配置文件副本
    os.system('copy ./fgsm_setting.json {}'.format(os.path.join(args_attack.global_settings.results_path, 'results{}/fgsm_setting.json'.format(args_attack.attacks.momentum))))
    print("实验配置已保存")
    
    #加载数据和模型
    attack_dataloader, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = prepare(init_all=True)
    
    #训练通用扰动
    for idx, (img_a, att_a, c_org) in enumerate(tqdm(attack_dataloader)):
        
        #早停
        if args_attack.global_settings.num_test is not None and idx * args_attack.global_settings.batch_size == args_attack.global_settings.num_test:
            break
        #加载图像和属性
        img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
        att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
        att_a = att_a.type(torch.float)
        if idx == 0:
            img_a_last = copy.deepcopy(img_a)
            
        # 攻击stargan
        solver.test_universal_model_level_attack(idx, img_a, c_org, fgsm_attack)
        
        # 攻击attentiongan
        attentiongan_solver.test_universal_model_level_attack(idx, img_a, c_org, fgsm_attack)
        
        # 攻击HiSD
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
        fgsm_attack.universal_perturb_HiSD(img_a.cuda(), transform, F, T, G, E, reference, x_trg+0.002, gen_models, mask)
        
        # 攻击AttGAN
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
            x_adv, perturb = fgsm_attack.universal_perturb_attgan(img_a, att_b_, gen_noattack, attgan)
        
        #保存扰动
        if fgsm_attack.method=='DI2_FGSM':
            torch.save(fgsm_attack.up,"./DI2_FGSM_perturbation.pt")
        elif fgsm_attack.method=='M_DI2_FGSM':
            torch.save(fgsm_attack.up,"./M_DI2_FGSM_perturbation.pt")
        print('save the Watermark')
    print('The size of Watermark: ', fgsm_attack.up.shape)

    #评估
    #evaluate_FGSM(args_attack, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models, fgsm_attack.up ,fgsm_attack.method)
    
#########################################

elif fgsm_attack.mode=='attgan':
    #创建目录
    target_dir = os.path.join(
        args_attack.AttGAN.result_dir,  # 父目录路径（可能不存在）
        'results{}'.format(args_attack.attacks.momentum)  # 子目录名
    )
    os.makedirs(target_dir, exist_ok=True)  # exist_ok=True 避免目录已存在时报错
    print("实验目录已创建")
    # 保存配置文件副本
    os.system('copy ./fgsm_setting.json {}'.format(
        os.path.join(args_attack.AttGAN.result_dir, 
        'results{}/fgsm_setting.json'.format(args_attack.attacks.momentum))
    ))
    print("实验配置已保存")
    
    #加载数据和模型
    attack_dataloader, test_dataloader, attgan, attgan_args= prepare(init_att=True)

    #为攻击器设置model
    fgsm_attack.model = attgan

    #攻击attgan模型
    for idx, (img_a, att_a, c_org) in enumerate(tqdm(attack_dataloader)):
        
        #早停
        if args_attack.global_settings.num_test is not None and idx * args_attack.global_settings.batch_size == args_attack.global_settings.num_test:#args_attack.global_settings.num_test=128  batch_size=8
            break
            
        #加载图像和属性
        img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
        att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
        att_a = att_a.type(torch.float)
        if idx == 0:
            img_a_last = copy.deepcopy(img_a)
        
        #攻击attgan
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
            x_adv, perturb = fgsm_attack.universal_perturb_attgan(img_a, att_b_, gen_noattack, attgan)
        
        #保存扰动
        if fgsm_attack.method=='DI2_FGSM':
            torch.save(fgsm_attack.att_up, args_attack.AttGAN.result_dir + "/DI2_FGSM_attgan_perturbation.pt")
        elif fgsm_attack.method=='M_DI2_FGSM':
            torch.save(fgsm_attack.att_up, args_attack.AttGAN.result_dir + "/M_DI2_FGSM_attgan_perturbation.pt")
        print('save the Watermark')
    print('The size of Watermark: ', fgsm_attack.up.shape)
        
        #评估
    #evaluate_AttGAN(args_attack, test_dataloader, attgan, attgan_args, fgsm_attack.att_up, fgsm_attack.method)
        
######################################
elif fgsm_attack.mode=='stargan':
    #创建目录
    target_dir = os.path.join(
        args_attack.stargan.result_dir,  # 父目录路径（可能不存在）
        'results{}'.format(args_attack.attacks.momentum)  # 子目录名
    )
    os.makedirs(target_dir, exist_ok=True)  # exist_ok=True 避免目录已存在时报错
    print("实验目录已创建")
    # 保存配置文件副本
    os.system('copy ./fgsm_setting.json {}'.format(
        os.path.join(args_attack.stargan.result_dir, 
        'results{}/fgsm_setting.json'.format(args_attack.attacks.momentum))
    ))
    print("实验配置已保存")
    
    #加载数据和模型
    attack_dataloader, test_dataloader,solver= prepare(init_star=True)

    #为攻击器设置model
    fgsm_attack.model = solver

    #攻击stargan模型
    for idx, (img_a, att_a, c_org) in enumerate(tqdm(attack_dataloader)):
        
        #早停
        if args_attack.global_settings.num_test is not None and idx * args_attack.global_settings.batch_size == args_attack.global_settings.num_test:#args_attack.global_settings.num_test=128  batch_size=8
            break
            
        #加载图像和属性
        img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
        att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
        att_a = att_a.type(torch.float)
        if idx == 0:
            img_a_last = copy.deepcopy(img_a)
            
        #攻击stargan
        solver.test_universal_model_level_attack(idx, img_a, c_org, fgsm_attack)
        
        #保存扰动
        if fgsm_attack.method=='DI2_FGSM':
            torch.save(fgsm_attack.star_up, args_attack.stargan.result_dir + "/DI2_FGSM_stargan_perturbation.pt")
        elif fgsm_attack.method=='M_DI2_FGSM':
            torch.save(fgsm_attack.star_up, args_attack.stargan.result_dir + "/M_DI2_FGSM_stargan_perturbation.pt")
        print('save the Watermark')
    print('The size of Watermark: ', fgsm_attack.star_up.shape)
        
    #评估
    #evaluate_stargan(args_attack, test_dataloader,solver, fgsm_attack.star_up, fgsm_attack.method)
        
################################
elif fgsm_attack.mode=='HiSD':
    #创建目录
    target_dir = os.path.join(
        args_attack.HiSD.result_dir,  # 父目录路径（可能不存在）
        'results{}'.format(args_attack.attacks.momentum)  # 子目录名
    )
    os.makedirs(target_dir, exist_ok=True)  # exist_ok=True 避免目录已存在时报错
    print("实验目录已创建")
    # 保存配置文件副本
    os.system('copy ./fgsm_setting.json {}'.format(
        os.path.join(args_attack.HiSD.result_dir, 
        'results{}/fgsm_setting.json'.format(args_attack.attacks.momentum))
    ))
    print("实验配置已保存")
    
    #加载数据和模型
    attack_dataloader, test_dataloader, transform, F, T, G, E, reference, gen_models = prepare(init_HISD=True)

    #为攻击器设置model
    fgsm_attack.model = gen_models

    #攻击HiSD模型
    for idx, (img_a, att_a, c_org) in enumerate(tqdm(attack_dataloader)):
        
        #早停
        if args_attack.global_settings.num_test is not None and idx * args_attack.global_settings.batch_size == args_attack.global_settings.num_test:#args_attack.global_settings.num_test=128  batch_size=8
            break
            
        #加载图像和属性
        img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
        att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
        att_a = att_a.type(torch.float)
        if idx == 0:
            img_a_last = copy.deepcopy(img_a)
            
        #攻击HiSD
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
        fgsm_attack.universal_perturb_HiSD(img_a.cuda(), transform, F, T, G, E, reference, x_trg+0.002, gen_models, mask)

        #保存扰动
        if fgsm_attack.method=='DI2_FGSM':
            torch.save(fgsm_attack.HISD_up, args_attack.HiSD.result_dir + "/DI2_FGSM_HiSD_perturbation.pt")
        elif fgsm_attack.method=='M_DI2_FGSM':
            torch.save(fgsm_attack.HISD_up, args_attack.HiSD.result_dir + "/M_DI2_FGSM_HiSD_perturbation.pt")
        print('save the Watermark')
    print('The size of Watermark: ', fgsm_attack.HISD_up.shape)
        
    #评估
    #evaluate_HIDF(args_attack, test_dataloader,transform, F, T, G, E, reference, gen_models, fgsm_attack.HISD_up, fgsm_attack.method)
        
################################
elif fgsm_attack.mode=='attention':
    #创建目录
    target_dir = os.path.join(
        args_attack.AttentionGAN.result_dir,  # 父目录路径（可能不存在）
        'results{}'.format(args_attack.attacks.momentum)  # 子目录名（如 results0.5）
    )
    os.makedirs(target_dir, exist_ok=True)  # exist_ok=True 避免目录已存在时报错
    print("实验目录已创建")
    # 保存配置文件副本
    os.system('copy ./fgsm_setting.json {}'.format(
        os.path.join(args_attack.AttentionGAN.result_dir, 
        'results{}/fgsm_setting.json'.format(args_attack.attacks.momentum))
    ))
    print("实验配置已保存")
    
    #加载数据和模型
    attack_dataloader, test_dataloader, attentiongan_solver= prepare(init_attention=True)
    
    #为攻击器设置model
    fgsm_attack.model = attentiongan_solver
    
    #攻击AttentionGAN模型
    for idx, (img_a, att_a, c_org) in enumerate(tqdm(attack_dataloader)):
        
        #早停
        if args_attack.global_settings.num_test is not None and idx * 16 == args_attack.global_settings.num_test:#args_attack.global_settings.num_test=128  batch_size=8
            break
            
        #加载图像和属性
        img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
        att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
        att_a = att_a.type(torch.float)
        if idx == 0:
            img_a_last = copy.deepcopy(img_a)
            
        #攻击attentiongan
        attentiongan_solver.test_universal_model_level_attack(idx, img_a, c_org, fgsm_attack)
        
        #保存扰动
        if fgsm_attack.method=='DI2_FGSM':
            torch.save(fgsm_attack.attention_up, args_attack.AttentionGAN.result_dir + "/DI2_FGSM_attention_perturbation.pt")
        elif fgsm_attack.method=='M_DI2_FGSM':
            torch.save(fgsm_attack.attention_up, args_attack.AttentionGAN.result_dir + "/M_DI2_FGSM_attention_perturbation.pt")
        print('save the Watermark')
    print('The size of Watermark: ', fgsm_attack.attention_up.shape)
    
    #评估
    #evaluate_AttentionGAN(args_attack, test_dataloader, attentiongan_solver, fgsm_attack.attention_up, fgsm_attack.method)