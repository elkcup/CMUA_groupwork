import torch
import torch.utils.data as data
import torchvision.utils as vutils
import torch.nn.functional as F
from torch import nn
import numpy as np
import os
from tqdm import tqdm
from AttGAN.data import check_attribute_conflict

def evaluate_pgd(args_attack, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference):
    '''评估PGD攻击的效果
    Args:
        args_attack: 攻击参数
        test_dataloader: 测试数据加载器
        attgan: AttGAN模型
        attgan_args: AttGAN参数
        solver: HiDF模型
        attentiongan_solver: AttentionGAN模型
        transform: 图像转换函数(未使用)
        F: 特征提取器
        T: 变换器
        G: 生成器
        E: 编码器
        reference: 参考图像
        '''

    device = torch.device('cuda' if torch.cuda.is_available() and args_attack.global_settings.gpu else 'cpu')
    print(f"使用设备: {device}")

    current_dir=os.path.dirname(os.path.abspath(__file__))
    perturb_path=current_dir
    
    perturb_attgan=torch.load(os.path.join(perturb_path,"pgd_attgan_perturbation.pt"))
    perturb_attention=torch.load(os.path.join(perturb_path,"pgd_attentiongan_perturbation.pt"))
    perturb_stargan=torch.load(os.path.join(perturb_path,"pgd_stargan_perturbation.pt"))
    perturb_hisd=torch.load(os.path.join(perturb_path,"pgd_hisd_perturbation.pt"))
    
    if perturb_stargan.size(0) > 1:
        perturb_stargan = perturb_stargan.mean(dim=0, keepdim=True)
    if perturb_attention.size(0) > 1:
        perturb_attention = perturb_attention.mean(dim=0, keepdim=True)
    if perturb_hisd.size(0) > 1:
        perturb_hisd = perturb_hisd.mean(dim=0, keepdim=True)
    if perturb_attgan.size(0) > 1:
        perturb_attgan = perturb_attgan.mean(dim=0, keepdim=True)

    l1_att,l2_att,min_dist_att,l0_att=0.0,0.0,0.0,0.0
    n_dist_att,n_samples_att=0,0

    l1_stargan,l2_stargan,min_dist_stargan,l0_stargan=0.0,0.0,0.0,0.0
    n_dist_stargan,n_samples_stargan=0,0

    l1_hisd,l2_hisd,min_dist_hisd,l0_hisd=0.0,0.0,0.0,0.0
    n_dist_hisd,n_samples_hisd=0,0

    l1_attention,l2_attention,min_dist_attention,l0_attention=0.0,0.0,0.0,0.0
    n_dist_attention,n_samples_attention=0,0


    for idx,(img_a,att_a,c_org) in enumerate(tqdm(test_dataloader, desc="PGD攻击进度")):
        if(args_attack.pgd_attacks.num_test is not None and idx>=args_attack.pgd_attacks.num_test):
            break
    ## 1. 评估HiSD
        img_a=img_a.to(device)
        
        with torch.no_grad():
            # 无扰动图像生成
            c = E(img_a)
            c_trg = c
            s_trg = F(reference, 1)
            c_trg = T(c_trg, s_trg, 1)
            gen_noattack = G(c_trg)
            # 有扰动图像生成
            c = E(img_a+perturb_hisd)
            c_trg = c
            s_trg = F(reference, 1)
            c_trg = T(c_trg, s_trg, 1)
            gen = G(c_trg)

            mask = torch.abs(gen_noattack - img_a)     
            mask = mask.sum(dim=1)                      
            mask = (mask > 0.5).float()                 
            mask_sum = mask.view(mask.size(0), -1).sum(dim=1) * 3 + 1e-8       
            diff = (gen * mask.unsqueeze(1) - gen_noattack * mask.unsqueeze(1)) ** 2  
            diff_sum = diff.view(diff.size(0), -1).sum(dim=1)  
            over_threshold = (diff_sum / mask_sum) > 0.05      
            n_dist_hisd += over_threshold.sum().item()
            n_samples_hisd += mask.size(0)

            l1_hisd += torch.nn.functional.l1_loss(gen, gen_noattack)
            l2_hisd += torch.nn.functional.mse_loss(gen, gen_noattack)
            l0_hisd += (gen - gen_noattack).norm(0)
            min_dist_hisd += (gen - gen_noattack).norm(float('-inf'))


    ## 2. 评估AttGAN
        att_a = att_a.type(torch.float).to(device)
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
                gen = attgan.G(img_a+perturb_attgan, att_b_)
                gen_noattack = attgan.G(img_a, att_b_)


            mask = torch.abs(gen_noattack - img_a)      
            mask = mask.sum(dim=1)                      
            mask = (mask > 0.5).float()                 
            mask_sum = mask.view(mask.size(0), -1).sum(dim=1) * 3 + 1e-8  
            diff = (gen * mask.unsqueeze(1) - gen_noattack * mask.unsqueeze(1)) ** 2  
            diff_sum = diff.view(diff.size(0), -1).sum(dim=1)  
            over_threshold = (diff_sum / mask_sum) > 0.05      
            n_dist_att += over_threshold.sum().item()
            n_samples_att += mask.size(0)

            l1_att += torch.nn.functional.l1_loss(gen, gen_noattack)
            l2_att += torch.nn.functional.mse_loss(gen, gen_noattack)
            l0_att += (gen - gen_noattack).norm(0)
            min_dist_att += (gen - gen_noattack).norm(float('-inf'))
     

    ## 3. 评估AttentionGAN
        x_noattack_list, x_fake_list = attentiongan_solver.test_universal_model_level(idx, img_a, c_org, perturb_attention, args_attack.AttentionGAN)
        for j in range(len(x_fake_list)):
            gen_noattack = x_noattack_list[j]
            gen = x_fake_list[j]
            
            mask = torch.abs(gen_noattack - img_a)      
            mask = mask.sum(dim=1)                      
            mask = (mask > 0.5).float()                 
            mask_sum = mask.view(mask.size(0), -1).sum(dim=1) * 3 + 1e-8  
            diff = (gen * mask.unsqueeze(1) - gen_noattack * mask.unsqueeze(1)) ** 2  
            diff_sum = diff.view(diff.size(0), -1).sum(dim=1)  
            over_threshold = (diff_sum / mask_sum) > 0.05      
            n_dist_attention += over_threshold.sum().item()
            n_samples_attention += mask.size(0)
        

            l1_attention += torch.nn.functional.l1_loss(gen, gen_noattack)
            l2_attention += torch.nn.functional.mse_loss(gen, gen_noattack)
            l0_attention += (gen - gen_noattack).norm(0)
            min_dist_attention += (gen - gen_noattack).norm(float('-inf'))

    ## 4. 评估StarGAN
        x_noattack_list, x_fake_list = solver.test_universal_model_level(idx, img_a, c_org, perturb_stargan, args_attack.stargan)
        for j in range(len(x_fake_list)):
            gen_noattack = x_noattack_list[j]
            gen = x_fake_list[j]

            mask = torch.abs(gen_noattack - img_a)      
            mask = mask.sum(dim=1)                      
            mask = (mask > 0.5).float()                 
            mask_sum = mask.view(mask.size(0), -1).sum(dim=1) * 3 + 1e-8  
            diff = (gen * mask.unsqueeze(1) - gen_noattack * mask.unsqueeze(1)) ** 2  
            diff_sum = diff.view(diff.size(0), -1).sum(dim=1)  
            over_threshold = (diff_sum / mask_sum) > 0.05      
            n_dist_stargan += over_threshold.sum().item()
            n_samples_stargan += mask.size(0)
            
            l1_stargan += torch.nn.functional.l1_loss(gen, gen_noattack)
            l2_stargan += torch.nn.functional.mse_loss(gen, gen_noattack)
            l0_stargan += (gen - gen_noattack).norm(0)
            min_dist_stargan += (gen - gen_noattack).norm(float('-inf'))

            if idx==0 and j==0:
                vutils.save_image(gen_noattack, 'outputs/PGD_gen_noattack.jpg', normalize=True, nrow=1)
                vutils.save_image(gen, 'outputs/PGD_gen_attack.jpg', normalize=True, nrow=1)
    
    # print(f"HiSD: l1 error: {l1_hisd / n_samples_hisd}, l2_error: {l2_hisd / n_samples_hisd}, prop_dist: {float(n_dist_hisd) / n_samples_hisd}, L0 error: {l0_hisd / n_samples_hisd}, L_-inf error: {min_dist_hisd / n_samples_hisd}.")
    hisd_prop_dist = float(n_dist_hisd) / n_samples_hisd
    
    # print(f"AttGAN: l1 error: {l1_att / n_samples_att}, l2_error: {l2_att / n_samples_att}, prop_dist: {float(n_dist_att) / n_samples_att}, L0 error: {l0_att / n_samples_att}, L_-inf error: {min_dist_att / n_samples_att}.")
    attgan_prop_dist = float(n_dist_att) / n_samples_att

    # print(f"AttentionGAN: l1 error: {l1_attention / n_samples_attention}, l2_error: {l2_attention / n_samples_attention}, prop_dist: {float(n_dist_attention) / n_samples_attention}, L0 error: {l0_attention / n_samples_attention}, L_-inf error: {min_dist_attention / n_samples_attention}.")
    attentiongan_prop_dist = float(n_dist_attention) / n_samples_attention

    # print(f"StarGAN: l1 error: {l1_stargan / n_samples_stargan}, l2_error: {l2_stargan / n_samples_stargan}, prop_dist: {float(n_dist_stargan) / n_samples_stargan}, L0 error: {l0_stargan / n_samples_stargan}, L_-inf error: {min_dist_stargan / n_samples_stargan}.")
    stargan_prop_dist = float(n_dist_stargan) / n_samples_stargan

    return hisd_prop_dist, attgan_prop_dist, attentiongan_prop_dist, stargan_prop_dist

def evaluate_mim(args_attack, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference):
    '''评估MIM攻击的效果
    Args:
        args_attack: 攻击参数
        test_dataloader: 测试数据加载器
        attgan: AttGAN模型
        attgan_args: AttGAN参数
        solver: HiDF模型
        attentiongan_solver: AttentionGAN模型
        transform: 图像转换函数
        F: 特征提取器
        T: 变换器
        G: 生成器
        E: 编码器
        reference: 参考图像
        '''

    device = torch.device('cuda' if torch.cuda.is_available() and args_attack.global_settings.gpu else 'cpu')
    print(f"使用设备: {device}")

    current_dir=os.path.dirname(os.path.abspath(__file__))
    purturb_path=current_dir
    
    perturb_attgan=torch.load(os.path.join(purturb_path,"mim_attgan_perturbation.pt"))
    perturb_attention=torch.load(os.path.join(purturb_path,"mim_attentiongan_perturbation.pt"))
    perturb_stargan=torch.load(os.path.join(purturb_path,"mim_stargan_perturbation.pt"))
    perturb_hisd=torch.load(os.path.join(purturb_path,"mim_hisd_perturbation.pt"))
    
    if perturb_stargan.size(0) > 1:
        perturb_stargan = perturb_stargan.mean(dim=0, keepdim=True) 
    if perturb_attention.size(0) > 1:
        perturb_attention = perturb_attention.mean(dim=0, keepdim=True)
    if perturb_hisd.size(0) > 1:
        perturb_hisd = perturb_hisd.mean(dim=0, keepdim=True)
    if perturb_attgan.size(0) > 1:
        perturb_attgan = perturb_attgan.mean(dim=0, keepdim=True)

    l1_att,l2_att,min_dist_att,l0_att=0.0,0.0,0.0,0.0
    n_dist_att,n_samples_att=0,0

    l1_stargan,l2_stargan,min_dist_stargan,l0_stargan=0.0,0.0,0.0,0.0
    n_dist_stargan,n_samples_stargan=0,0

    l1_hisd,l2_hisd,min_dist_hisd,l0_hisd=0.0,0.0,0.0,0.0
    n_dist_hisd,n_samples_hisd=0,0

    l1_attention,l2_attention,min_dist_attention,l0_attention=0.0,0.0,0.0,0.0
    n_dist_attention,n_samples_attention=0,0


    for idx,(img_a,att_a,c_org) in enumerate(tqdm(test_dataloader, desc="MIM攻击进度")):
        if(args_attack.pgd_attacks.num_test is not None and idx>=args_attack.pgd_attacks.num_test):
            break
    ## 1. 评估HiSD
        img_a=img_a.to(device)
        
        with torch.no_grad():
            # 无扰动图像生成
            c = E(img_a)
            c_trg = c
            s_trg = F(reference, 1)
            c_trg = T(c_trg, s_trg, 1)
            gen_noattack = G(c_trg)
            # 有扰动图像生成
            c = E(img_a+perturb_hisd)
            c_trg = c
            s_trg = F(reference, 1)
            c_trg = T(c_trg, s_trg, 1)
            gen = G(c_trg)

            mask = torch.abs(gen_noattack - img_a)      
            mask = mask.sum(dim=1)                      
            mask = (mask > 0.5).float()                 
            mask_sum = mask.view(mask.size(0), -1).sum(dim=1) * 3 + 1e-8  
            diff = (gen * mask.unsqueeze(1) - gen_noattack * mask.unsqueeze(1)) ** 2  
            diff_sum = diff.view(diff.size(0), -1).sum(dim=1) 
            over_threshold = (diff_sum / mask_sum) > 0.05      
            n_dist_hisd += over_threshold.sum().item()
            n_samples_hisd += mask.size(0)

            l1_hisd += torch.nn.functional.l1_loss(gen, gen_noattack)
            l2_hisd += torch.nn.functional.mse_loss(gen, gen_noattack)
            l0_hisd += (gen - gen_noattack).norm(0)
            min_dist_hisd += (gen - gen_noattack).norm(float('-inf'))


    ## 2. 评估AttGAN
        att_a = att_a.type(torch.float).to(device)
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
                gen = attgan.G(img_a+perturb_attgan, att_b_)
                gen_noattack = attgan.G(img_a, att_b_)


            mask = torch.abs(gen_noattack - img_a)      
            mask = mask.sum(dim=1)                      
            mask = (mask > 0.5).float()                
            mask_sum = mask.view(mask.size(0), -1).sum(dim=1) * 3 + 1e-8  
            diff = (gen * mask.unsqueeze(1) - gen_noattack * mask.unsqueeze(1)) ** 2  
            diff_sum = diff.view(diff.size(0), -1).sum(dim=1) 
            over_threshold = (diff_sum / mask_sum) > 0.05      
            n_dist_att += over_threshold.sum().item()
            n_samples_att += mask.size(0)

            l1_att += torch.nn.functional.l1_loss(gen, gen_noattack)
            l2_att += torch.nn.functional.mse_loss(gen, gen_noattack)
            l0_att += (gen - gen_noattack).norm(0)
            min_dist_att += (gen - gen_noattack).norm(float('-inf'))
     

    ## 3. 评估AttentionGAN
        x_noattack_list, x_fake_list = attentiongan_solver.test_universal_model_level(idx, img_a, c_org, perturb_attention, args_attack.AttentionGAN)
        for j in range(len(x_fake_list)):
            gen_noattack = x_noattack_list[j]
            gen = x_fake_list[j]
            
            mask = torch.abs(gen_noattack - img_a)     
            mask = mask.sum(dim=1)                      
            mask = (mask > 0.5).float()                 
            mask_sum = mask.view(mask.size(0), -1).sum(dim=1) * 3 + 1e-8  
            diff = (gen * mask.unsqueeze(1) - gen_noattack * mask.unsqueeze(1)) ** 2 
            diff_sum = diff.view(diff.size(0), -1).sum(dim=1)  
            over_threshold = (diff_sum / mask_sum) > 0.05     
            n_dist_attention += over_threshold.sum().item()
            n_samples_attention += mask.size(0)
        

            l1_attention += torch.nn.functional.l1_loss(gen, gen_noattack)
            l2_attention += torch.nn.functional.mse_loss(gen, gen_noattack)
            l0_attention += (gen - gen_noattack).norm(0)
            min_dist_attention += (gen - gen_noattack).norm(float('-inf'))

    ## 4. 评估StarGAN
        x_noattack_list, x_fake_list = solver.test_universal_model_level(idx, img_a, c_org, perturb_stargan, args_attack.stargan)
        for j in range(len(x_fake_list)):
            gen_noattack = x_noattack_list[j]
            gen = x_fake_list[j]

            mask = torch.abs(gen_noattack - img_a)      
            mask = mask.sum(dim=1)                      
            mask = (mask > 0.5).float()                 
            mask_sum = mask.view(mask.size(0), -1).sum(dim=1) * 3 + 1e-8  
            diff = (gen * mask.unsqueeze(1) - gen_noattack * mask.unsqueeze(1)) ** 2 
            diff_sum = diff.view(diff.size(0), -1).sum(dim=1)  
            over_threshold = (diff_sum / mask_sum) > 0.05     
            n_dist_stargan += over_threshold.sum().item()
            n_samples_stargan += mask.size(0)
            
            l1_stargan += torch.nn.functional.l1_loss(gen, gen_noattack)
            l2_stargan += torch.nn.functional.mse_loss(gen, gen_noattack)
            l0_stargan += (gen - gen_noattack).norm(0)
            min_dist_stargan += (gen - gen_noattack).norm(float('-inf'))
            if idx==0 and j==0:
                vutils.save_image(gen_noattack, 'outputs/MIM_gen_noattack.jpg', normalize=True, nrow=1)
                vutils.save_image(gen, 'outputs/MIM_gen_attack.jpg', normalize=True, nrow=1)
                vutils.save_image(img_a+perturb_stargan, 'outputs/MIM_img_a.jpg', normalize=True, nrow=1)
    
    # print(f"HiSD: l1 error: {l1_hisd / n_samples_hisd}, l2_error: {l2_hisd / n_samples_hisd}, prop_dist: {float(n_dist_hisd) / n_samples_hisd}, L0 error: {l0_hisd / n_samples_hisd}, L_-inf error: {min_dist_hisd / n_samples_hisd}.")
    hisd_prop_dist = float(n_dist_hisd) / n_samples_hisd
    
    # print(f"AttGAN: l1 error: {l1_att / n_samples_att}, l2_error: {l2_att / n_samples_att}, prop_dist: {float(n_dist_att) / n_samples_att}, L0 error: {l0_att / n_samples_att}, L_-inf error: {min_dist_att / n_samples_att}.")
    attgan_prop_dist = float(n_dist_att) / n_samples_att

    # print(f"AttentionGAN: l1 error: {l1_attention / n_samples_attention}, l2_error: {l2_attention / n_samples_attention}, prop_dist: {float(n_dist_attention) / n_samples_attention}, L0 error: {l0_attention / n_samples_attention}, L_-inf error: {min_dist_attention / n_samples_attention}.")
    attentiongan_prop_dist = float(n_dist_attention) / n_samples_attention

    # print(f"StarGAN: l1 error: {l1_stargan / n_samples_stargan}, l2_error: {l2_stargan / n_samples_stargan}, prop_dist: {float(n_dist_stargan) / n_samples_stargan}, L0 error: {l0_stargan / n_samples_stargan}, L_-inf error: {min_dist_stargan / n_samples_stargan}.")
    stargan_prop_dist = float(n_dist_stargan) / n_samples_stargan

    return hisd_prop_dist, attgan_prop_dist, attentiongan_prop_dist, stargan_prop_dist
