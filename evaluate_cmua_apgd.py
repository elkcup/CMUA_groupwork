from torch import nn
import numpy as np
import torch
import torch.utils.data as data

import torchvision.utils as vutils
import torch.nn.functional as F

import argparse
import copy
import json
import os
from os.path import join
import sys
import matplotlib.image
from tqdm import tqdm

from AttGAN.data import check_attribute_conflict
import torchvision.utils as vutils


from data import CelebA

from model_data_prepare import prepare

def evaluate_multiple_models(args_attack, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models, water,method):
    
        
    #  HiDF inference and evaluating
    l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
    n_dist, n_samples = 0, 0
    for idx, (img_a, att_a, c_org) in enumerate(test_dataloader):
        # if args_attack.global_settings.num_test is not None and idx == args_attack.global_settings.num_test * args_attack.global_settings.batch_size:
        #     break

        # img_a = gauss

        img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
        
        if idx ==0:
            vutils.save_image(img_a,'outputs/original.jpg',nrow=1, normalize=True)
        
        if idx==0 and method == 'CMUA':
            vutils.save_image(img_a+water,'outputs/CMUA_perturb.jpg',nrow=1, normalize=True)
        elif idx==0 and method=='APGD':
            vutils.save_image(img_a+water,'outputs/APGD_perturb.jpg',nrow=1, normalize=True)
        
        with torch.no_grad():
            # clean
            c = E(img_a)
            c_trg = c
            s_trg = F(reference, 1)
            c_trg = T(c_trg, s_trg, 1)
            gen_noattack = G(c_trg)
            # adv
            c = E(img_a + water)
            c_trg = c
            s_trg = F(reference, 1)
            c_trg = T(c_trg, s_trg, 1)
            gen = G(c_trg)
            mask = abs(gen_noattack - img_a)
            mask = mask[0,0,:,:] + mask[0,1,:,:] + mask[0,2,:,:]
            mask[mask>0.5] = 1
            mask[mask<0.5] = 0

            l1_error += torch.nn.functional.l1_loss(gen, gen_noattack)
            l2_error += torch.nn.functional.mse_loss(gen, gen_noattack)
            l0_error += (gen - gen_noattack).norm(0)
            min_dist += (gen - gen_noattack).norm(float('-inf'))
            if (((gen*mask - gen_noattack*mask)**2).sum() / (mask.sum()*3)) > 0.05:
                n_dist += 1
            n_samples += 1

    HiDF_prop_dist = float(n_dist) / n_samples

    # AttGAN inference and evaluating
    l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
    n_dist, n_samples = 0, 0
    for idx, (img_a, att_a, c_org) in enumerate(test_dataloader):
        # if args_attack.global_settings.num_test is not None and idx == args_attack.global_settings.num_test * args_attack.global_settings.batch_size:
        #     break
        img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
        att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
        att_a = att_a.type(torch.float)   
        att_b_list = [att_a]
        for i in range(attgan_args.n_attrs):
            tmp = att_a.clone()
            tmp[:, i] = 1 - tmp[:, i]
            tmp = check_attribute_conflict(tmp, attgan_args.attrs[i], attgan_args.attrs)
            att_b_list.append(tmp)
        samples = [img_a, img_a+water]
        noattack_list = []
        for i, att_b in enumerate(att_b_list):
            att_b_ = (att_b * 2 - 1) * attgan_args.thres_int
            if i > 0:
                att_b_[..., i - 1] = att_b_[..., i - 1] * attgan_args.test_int / attgan_args.thres_int
            with torch.no_grad():
                gen = attgan.G(img_a+water, att_b_)
                gen_noattack = attgan.G(img_a, att_b_)
    
            samples.append(gen)
            noattack_list.append(gen_noattack)

            mask = abs(gen_noattack - img_a)
            mask = mask[0,0,:,:] + mask[0,1,:,:] + mask[0,2,:,:]
            mask[mask>0.5] = 1
            mask[mask<0.5] = 0

            l1_error += torch.nn.functional.l1_loss(gen, gen_noattack)
            l2_error += torch.nn.functional.mse_loss(gen, gen_noattack)
            l0_error += (gen - gen_noattack).norm(0)
            min_dist += (gen - gen_noattack).norm(float('-inf'))
            if (((gen*mask - gen_noattack*mask)**2).sum() / (mask.sum()*3)) > 0.05:
                n_dist += 1
            n_samples += 1
    
    attgan_prop_dist = float(n_dist) / n_samples

    # stargan inference and evaluating
    l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
    n_dist, n_samples = 0, 0
    for idx, (img_a, att_a, c_org) in enumerate(test_dataloader):
        # if args_attack.global_settings.num_test is not None and idx == args_attack.global_settings.num_test * args_attack.global_settings.batch_size:
        #     break
        img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
        att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
        att_a = att_a.type(torch.float)
        x_noattack_list, x_fake_list = solver.test_universal_model_level(idx, img_a, c_org, water, args_attack.stargan)
        for j in range(len(x_fake_list)):
            gen_noattack = x_noattack_list[j]
            gen = x_fake_list[j]
            
            if idx==0 and j==0 and method == 'CMUA':
                vutils.save_image(gen_noattack,'outputs/CMUA_gen_noattack.jpg',nrow=1, normalize=True)
                vutils.save_image(gen,'outputs/CMUA_gen_attack.jpg',nrow=1, normalize=True)
            elif idx==0 and j==0 and method =='APGD':
                vutils.save_image(gen_noattack,'outputs/APGD_gen_noattack.jpg',nrow=1, normalize=True)
                vutils.save_image(gen,'outputs/APGD_gen_attack.jpg',nrow=1, normalize=True)
            
            mask = abs(gen_noattack - img_a)
            mask = mask[0,0,:,:] + mask[0,1,:,:] + mask[0,2,:,:]
            mask[mask>0.5] = 1
            mask[mask<0.5] = 0

            l1_error += torch.nn.functional.l1_loss(gen, gen_noattack)
            l2_error += torch.nn.functional.mse_loss(gen, gen_noattack)
            l0_error += (gen - gen_noattack).norm(0)
            min_dist += (gen - gen_noattack).norm(float('-inf'))
            if (((gen*mask - gen_noattack*mask)**2).sum() / (mask.sum()*3)) > 0.05:
                n_dist += 1
            n_samples += 1
        
    stargan_prop_dist = float(n_dist) / n_samples

    # AttentionGAN inference and evaluating
    l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
    n_dist, n_samples = 0, 0
    for idx, (img_a, att_a, c_org) in enumerate(test_dataloader):
        # if args_attack.global_settings.num_test is not None and idx == args_attack.global_settings.num_test * args_attack.global_settings.batch_size:
        #     break
        img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
        att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
        att_a = att_a.type(torch.float)
        x_noattack_list, x_fake_list = attentiongan_solver.test_universal_model_level(idx, img_a, c_org, water, args_attack.AttentionGAN)
        for j in range(len(x_fake_list)):
            gen_noattack = x_noattack_list[j]
            gen = x_fake_list[j]
            mask = abs(gen_noattack - img_a)
            mask = mask[0,0,:,:] + mask[0,1,:,:] + mask[0,2,:,:]
            mask[mask>0.5] = 1
            mask[mask<0.5] = 0

            l1_error += torch.nn.functional.l1_loss(gen, gen_noattack)
            l2_error += torch.nn.functional.mse_loss(gen, gen_noattack)
            l0_error += (gen - gen_noattack).norm(0)
            min_dist += (gen - gen_noattack).norm(float('-inf'))
            if (((gen*mask - gen_noattack*mask)**2).sum() / (mask.sum()*3)) > 0.05:
                n_dist += 1
            n_samples += 1
        

    aggan_prop_dist = float(n_dist) / n_samples
    return HiDF_prop_dist,  attgan_prop_dist, aggan_prop_dist,stargan_prop_dist

def evaluate_apgd(args_attack, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models):
    # load the trained CMUA-Watermark
    water = torch.load('./auto_pgd_perturbation.pt')

    # Init the attacked models

    return evaluate_multiple_models(args_attack, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models, water,'APGD')


def evaluate_cmua(args_attack, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models):
    # load the trained CMUA-Watermark
    water = torch.load('./perturbation.pt')

    # Init the attacked models

    return evaluate_multiple_models(args_attack, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models, water,'CMUA')