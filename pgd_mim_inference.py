# 注：此处的推理工作与其他部分高度重合，因而直接套用的其它部分代码
import argparse
import copy
import json
import os
import sys
from os.path import join

import matplotlib.image
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.utils as vutils
from tqdm import tqdm

from evaluate_pgd_mim import evaluate_pgd,evaluate_mim
from AttGAN.data import check_attribute_conflict
from data import CelebA
from model_data_prepare import prepare


class ObjDict(dict):
    """
    Makes a  dictionary behave like an object,with attribute-style access.
    """
    def __getattr__(self,name):
        try:
            return self[name]
        except:
            raise AttributeError(name)
    def __setattr__(self,name,value):
        self[name]=value

def parse(args=None):
    with open(join('./setting.json'), 'r') as f:
        args_attack = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
    return args_attack

# 保存推理结果 
current_path = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(current_path, 'inference_results')
pgd_path = os.path.join(results_path, 'pgd')
mim_path = os.path.join(results_path, 'mim')
if not os.path.exists(pgd_path):
    os.makedirs(pgd_path)
if not os.path.exists(mim_path):
    os.makedirs(mim_path)

args_attack = parse()

attack_dataloader, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = prepare()

evaluate_pgd(args_attack, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, pgd_path)
evaluate_mim(args_attack, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, mim_path)