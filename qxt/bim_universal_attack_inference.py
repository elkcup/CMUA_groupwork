# bim_universal_attack_inference.py
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


class ObjDict(dict):
    def __getattr__(self,name):
        try:
            return self[name]
        except:
            raise AttributeError(f"'ObjDict' 没有该属性 '{name}'")
    
    def __setattr__(self,name,value):
        self[name]=value

def parse(args=None):
    with open(join('./bim_setting.json'), 'r') as f:
        args_attack = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

        
    return args_attack


args_attack = parse()
print(args_attack)
os.system('copy -r ./results {}/results{}'.format(args_attack.global_settings.results_path, args_attack.attacks.momentum))
print("已创建实验目录")
os.system('copy ./bim_setting.json {}'.format(os.path.join(args_attack.global_settings.results_path, 'results{}/bim_setting.json'.format(args_attack.attacks.momentum))))
print("实验配置已保存")

# 初始化
def init_Attack(args_attack):
    bim_attack = attacks.BIMAttack(model=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), epsilon=args_attack.attacks.epsilon, k=args_attack.attacks.k, a=args_attack.attacks.a, star_factor=args_attack.attacks.star_factor, attention_factor=args_attack.attacks.attention_factor, att_factor=args_attack.attacks.att_factor, HiSD_factor=args_attack.attacks.HiSD_factor, args=args_attack.attacks)
    return bim_attack


bim_attack = init_Attack(args_attack)

# 加载训练好的CMUA-水印
if args_attack.global_settings.universal_perturbation_path:
    bim_attack.up = torch.load(args_attack.global_settings.universal_perturbation_path)


# 初始化被攻击模型
attack_dataloader, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = prepare()
print("已完成初始化攻击模型")


print('CMUA-水印的大小: ', bim_attack.up.shape)
evaluate_multiple_models(args_attack, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models, bim_attack)