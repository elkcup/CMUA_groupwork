#  bim_universal_attack.py
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
import bim_attack as attacks

from model_data_prepare import prepare
from bim_evaluate import evaluate_multiple_models

class ObjDict(dict):  # 使字典像对象一样，具有属性式访问
    def __getattr__(self,name):  # 当尝试访问一个不存在的属性时，__getattr__ 方法会被调用。
        try:
            return self[name]
        except:
            raise AttributeError(f"'ObjDict' 没有该属性 '{name}'")
    
    def __setattr__(self,name,value):  # 当尝试设置一个属性时，__setattr__ 方法会被调用
        self[name]=value

def parse(args=None):
    with open(join('./bim_setting.json'), 'r') as f:
        args_attack = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

        
    return args_attack


args_attack = parse()
print(args_attack)
os.system('copy -r ./results {}/results{}'.format(args_attack.global_settings.results_path, args_attack.attacks.momentum))
print("已创建实验目录")
os.system('copy ./bim_setting.json {}'.format(os.path.join(args_attack.global_settings.results_path, 'results{}/setting.json'.format(args_attack.attacks.momentum))))
print("实验配置已保存")

# 初始化
def init_Attack(args_attack):
    bim_attack = attacks.BIMAttack(model=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), epsilon=args_attack.attacks.epsilon, k=args_attack.attacks.k, a=args_attack.attacks.a, star_factor=args_attack.attacks.star_factor, attention_factor=args_attack.attacks.attention_factor, att_factor=args_attack.attacks.att_factor, HiSD_factor=args_attack.attacks.HiSD_factor, args=args_attack.attacks)
    return bim_attack


bim_attack = init_Attack(args_attack)

# 载入已有扰动
# if args_attack.global_settings.universal_perturbation_path:
#     pgd_attack.up = torch.load(args_attack.global_settings.universal_perturbation_path)


# 初始化被攻击模型
attack_dataloader, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = prepare()
print("已完成初始化攻击模型，仅攻击2个周期")

# 攻击模型
for i in range(1):
    for idx, (img_a, att_a, c_org) in enumerate(tqdm(attack_dataloader)):
        if args_attack.global_settings.num_test is not None and idx * args_attack.global_settings.batch_size == args_attack.global_settings.num_test:
            break
            """
            attack_dataloader攻击数据加载器，用于加载攻击所需的数据
            args_attack包含攻击配置的参数对象
            num_test测试样本数量
            batch_size批量大小
            """   
        img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
        att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
        att_a = att_a.type(torch.float)

        if idx == 0:
            img_a_last = copy.deepcopy(img_a)

        # 攻击 stargan
        solver.test_universal_model_level_attack(idx, img_a, c_org, bim_attack)

        # 攻击 attentiongan
        attentiongan_solver.test_universal_model_level_attack(idx, img_a, c_org, bim_attack)

        # 攻击 HiSD
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
        bim_attack.universal_perturb_HiSD(img_a.cuda(), transform, F, T, G, E, reference, x_trg+0.002, gen_models, mask)

        # 攻击 AttGAN
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
            x_adv, perturb = bim_attack.universal_perturb_attgan(img_a, att_b_, gen_noattack, attgan)

        torch.save(bim_attack.up, args_attack.global_settings.universal_perturbation_path)
        print('保存CMUA-水印')

        

print('CMUA-水印的大小: ', bim_attack.up.shape)
evaluate_multiple_models(args_attack, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models, bim_attack)
