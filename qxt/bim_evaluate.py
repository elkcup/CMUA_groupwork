# bim_evaluate.py
import torch
import torch.utils.data as data
import torchvision.utils as vutils
import torch.nn.functional as F
from torch import nn
import numpy as np
import bim_attack as attacks

from AttGAN.data import check_attribute_conflict

class GaussianBlurConv(nn.Module):  # 高斯模糊卷积
    def __init__(self, channels=3):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]  # 5x5高斯核
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)  # 将高斯核转换为张量，并增加必要的维度，使其形状为 [1, 1, 5, 5]
        kernel = np.repeat(kernel, self.channels, axis=0)  # 复制到3通道 [3,1,5,5]
        self.weight = nn.Parameter(data=kernel, requires_grad=False)  # 固定参数 

    def __call__(self, x):
        x = F.conv2d(x, self.weight, padding=2, groups=self.channels)  # 分组卷积实现逐通道模糊
        return x

def evaluate_multiple_models(args_attack, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models, bim_attack):

    #  HiDF推断和评估
    l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
    n_dist, n_samples = 0, 0
    
    for idx, (img_a, att_a, c_org) in enumerate(test_dataloader):
        # if args_attack.global_settings.num_test is not None and idx == args_attack.global_settings.num_test * args_attack.global_settings.batch_size:
        #     break

        # img_a = gauss
        if idx == 20:
            break
        
        img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
        
        with torch.no_grad():
            # 干净样本生成
            c = E(img_a)  # 编码器提取特征
            c_trg = c  # 目标特征
            s_trg = F(reference, 1)  # 风格提取
            c_trg = T(c_trg, s_trg, 1)   # 特征转换
            gen_noattack = G(c_trg)  # 生成器产生干净样本
            
            # 对抗样本生成
            c = E(img_a + bim_attack.up)  # 添加扰动后的编码
            c_trg = c
            s_trg = F(reference, 1)
            c_trg = T(c_trg, s_trg, 1)
            gen = G(c_trg)
            
            # 计算差异mask
            mask = abs(gen_noattack - img_a)
            mask = mask[0,0,:,:] + mask[0,1,:,:] + mask[0,2,:,:]
            mask[mask>0.5] = 1
            mask[mask<0.5] = 0

            # 误差计算
            l1_error += torch.nn.functional.l1_loss(gen, gen_noattack)
            l2_error += torch.nn.functional.mse_loss(gen, gen_noattack)
            l0_error += (gen - gen_noattack).norm(0)
            min_dist += (gen - gen_noattack).norm(float('-inf'))
            if (((gen*mask - gen_noattack*mask)**2).sum() / (mask.sum()*3)) > 0.05:
                n_dist += 1
            n_samples += 1

    print('HiDF {} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))
    HiDF_results = 'HiDF {} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples)
    HiDF_prop_dist = float(n_dist) / n_samples

    # AttGAN推理和评估
    l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
    n_dist, n_samples = 0, 0
    for idx, (img_a, att_a, c_org) in enumerate(test_dataloader):
        # if args_attack.global_settings.num_test is not None and idx == args_attack.global_settings.num_test * args_attack.global_settings.batch_size:
        #     break
        if idx == 20:
            break
        img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
        att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
        att_a = att_a.type(torch.float)   
        att_b_list = [att_a]
        for i in range(attgan_args.n_attrs):
            tmp = att_a.clone()
            tmp[:, i] = 1 - tmp[:, i]
            tmp = check_attribute_conflict(tmp, attgan_args.attrs[i], attgan_args.attrs)
            att_b_list.append(tmp)
        samples = [img_a, img_a+bim_attack.up]
        noattack_list = []
        for i, att_b in enumerate(att_b_list):
            att_b_ = (att_b * 2 - 1) * attgan_args.thres_int
            if i > 0:
                att_b_[..., i - 1] = att_b_[..., i - 1] * attgan_args.test_int / attgan_args.thres_int
            with torch.no_grad():
                gen = attgan.G(img_a+bim_attack.up, att_b_)
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
        
      
        
        
    print('AttGAN {} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))
    AttGAN_results = 'AttGAN {} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples)
    attgan_prop_dist = float(n_dist) / n_samples

    # stargan推理和评估
    l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
    n_dist, n_samples = 0, 0
    for idx, (img_a, att_a, c_org) in enumerate(test_dataloader):
        # if args_attack.global_settings.num_test is not None and idx == args_attack.global_settings.num_test * args_attack.global_settings.batch_size:
        #     break
        if idx == 20:
            break
        img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
        att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
        att_a = att_a.type(torch.float)
        x_noattack_list, x_fake_list = solver.test_universal_model_level(idx, img_a, c_org, bim_attack.up, args_attack.stargan)
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
        

    print('stargan {} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))
    stargan_results = 'stargan {} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples)
    stargan_prop_dist = float(n_dist) / n_samples

    # AttentionGAN推理和评估
    l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
    n_dist, n_samples = 0, 0
    for idx, (img_a, att_a, c_org) in enumerate(test_dataloader):
        # if args_attack.global_settings.num_test is not None and idx == args_attack.global_settings.num_test * args_attack.global_settings.batch_size:
        #     break
        if idx == 20:
            break
        img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
        att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
        att_a = att_a.type(torch.float)
        x_noattack_list, x_fake_list = attentiongan_solver.test_universal_model_level(idx, img_a, c_org, bim_attack.up, args_attack.AttentionGAN)
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
        

    print('attentiongan {} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))
    AttentionGAN_results = 'attentiongan {} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples)
    aggan_prop_dist = float(n_dist) / n_samples
    return HiDF_prop_dist, stargan_prop_dist, attgan_prop_dist, aggan_prop_dist


def evaluate_bim(args_attack, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models):
    # 初始化 BIMAttack 实例
    bim_attack = attacks.BIMAttack(model=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), epsilon=args_attack.attacks.epsilon, k=args_attack.attacks.k, a=args_attack.attacks.a, star_factor=args_attack.attacks.star_factor, attention_factor=args_attack.attacks.attention_factor, att_factor=args_attack.attacks.att_factor, HiSD_factor=args_attack.attacks.HiSD_factor, args=args_attack.attacks)
    
    # 加载扰动
    if args_attack.global_settings.universal_perturbation_path:
        bim_attack.up = torch.load(args_attack.global_settings.universal_perturbation_path)
    else:
        bim_attack.up = torch.zeros_like(test_dataloader.dataset[0][0]).unsqueeze(0).to(bim_attack.device)
    
    return evaluate_multiple_models(args_attack, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models, bim_attack)
