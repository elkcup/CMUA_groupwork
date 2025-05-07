import torch
import torch.utils.data as data
import torchvision.utils as vutils
import torch.nn.functional as F
from torch import nn
import numpy as np

from AttGAN.data import check_attribute_conflict

#attgan专用扰动评估
def evaluate_AttGAN(args_attack, test_dataloader, attgan, attgan_args, up, method):
    
    #n_dist 统计显著变化样本数，n_samples 统计总样本数
    n_dist, n_samples = 0, 0
    
    #遍历测试数据
    for idx, (img_a, att_a, c_org) in enumerate(test_dataloader):
        
        #加载图像和属性
        img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
        att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
        att_a = att_a.type(torch.float)   
        
        #对每个属性进行反转并检查冲突（例如在属性反转后由“秃头”变成了“不秃头”，则此时需要检测“刘海”），并添加到属性列表att_b_list中
        att_b_list = [att_a]
        for i in range(attgan_args.n_attrs):
            tmp = att_a.clone()
            tmp[:, i] = 1 - tmp[:, i]
            tmp = check_attribute_conflict(tmp, attgan_args.attrs[i], attgan_args.attrs)
            att_b_list.append(tmp)
        
        #samples用于存储添加了扰动的样本，noattack_list用于存储没有扰动的样本
        samples = [img_a, img_a+up]
        noattack_list = []
        
        #遍历每个属性
        for i, att_b in enumerate(att_b_list):
            
            #将属性值缩放到 [-thres_int, thres_int] 范围(归一化)
            att_b_ = (att_b * 2 - 1) * attgan_args.thres_int
            if i > 0:
                att_b_[..., i - 1] = att_b_[..., i - 1] * attgan_args.test_int / attgan_args.thres_int
                
            #使用生成器 attgan.G 生成对抗样本 gen（输入为扰动后的图像）和无攻击样本 gen_noattack
            with torch.no_grad():
                gen = attgan.G(img_a+up, att_b_)
                gen_noattack = attgan.G(img_a, att_b_)
            samples.append(gen)
            noattack_list.append(gen_noattack)
            
            #计算生成图像（无攻击）与原始图像的绝对差异（mask），将差异在三个通道上求和，得到单通道掩码，并二值化（阈值 0.5）
            mask = abs(gen_noattack - img_a)
            mask = mask[0,0,:,:] + mask[0,1,:,:] + mask[0,2,:,:]
            mask[mask>0.5] = 1
            mask[mask<0.5] = 0
            
            #统计显著变化的样本并更新数量（用于计算L2_mask）
            if (((gen*mask - gen_noattack*mask)**2).sum() / (mask.sum()*3)) > 0.05:
                n_dist += 1
            n_samples += 1
        
        #保存图片(根据需要保存) 
        #for j in range(len(samples)-2):
        #    
        #    # 保存对抗样本生成的图片
        #    out_file = './demo_results/AttGAN_advgen_{}.jpg'.format(j)
        #    vutils.save_image(samples[j+2], out_file, nrow=1, normalize=True, value_range=(-1., 1.))
        #    
        #    # 保存原图生成的图片
        #    out_file = './demo_results/AttGAN_gen_{}.jpg'.format(j)
        #    vutils.save_image(noattack_list[j], out_file, nrow=1, normalize=True, value_range=(-1., 1.))
        
    #计算AttGAN L2_mask
    attgan_prop_dist = float(n_dist) / n_samples
    
    return attgan_prop_dist


def evaluate_stargan(args_attack, test_dataloader,solver, up, method):
    
    #初始化
    n_dist, n_samples = 0, 0
    
    #遍历测试数据
    for idx, (img_a, att_a, c_org) in enumerate(test_dataloader):
        
        #加载图像和属性
        img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
        att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
        att_a = att_a.type(torch.float)
        
        #生成对抗样本和无攻击样本
        x_noattack_list, x_fake_list = solver.test_universal_model_level(
            idx, img_a, c_org, up, args_attack.stargan)
        
        #评估
        for j in range(len(x_fake_list)):
            gen_noattack = x_noattack_list[j]
            gen = x_fake_list[j]
            mask = abs(gen_noattack - img_a)
            mask = mask[0,0,:,:] + mask[0,1,:,:] + mask[0,2,:,:]
            mask[mask>0.5] = 1
            mask[mask<0.5] = 0
            if (((gen*mask - gen_noattack*mask)**2).sum() / (mask.sum()*3)) > 0.05:
                n_dist += 1
            n_samples += 1
            if idx==0 and j==0:
                if method=='DI2_FGSM':
                    #保存添加水印后的图片
                    vutils.save_image(img_a + up,'outputs/DI2_FGSM_perturb.jpg',nrow=1, normalize=True)
                    # 保存原图生成图片
                    gen_noattack = x_noattack_list[j]
                    vutils.save_image(gen_noattack, 'outputs/DI2_FGSM_gen_noattack.jpg', nrow=1, normalize=True, value_range=(-1., 1.))
                    # 保存对抗样本生成图片
                    gen = x_fake_list[j]
                    vutils.save_image(gen, 'outputs/DI2_FGSM_gen_attack.jpg', nrow=1, normalize=True, value_range=(-1., 1.)) 
                elif method=='M_DI2_FGSM':
                    vutils.save_image(img_a + up,'outputs/M_DI2_FGSM_perturb.jpg',nrow=1, normalize=True)
                    # 保存原图生成图片
                    gen_noattack = x_noattack_list[j]
                    vutils.save_image(gen_noattack, 'outputs/M_DI2_FGSM_gen_noattack.jpg', nrow=1, normalize=True, value_range=(-1., 1.))
                    # 保存对抗样本生成图片
                    gen = x_fake_list[j]
                    vutils.save_image(gen, 'outputs/M_DI2_FGSM_gen_attack.jpg', nrow=1, normalize=True, value_range=(-1., 1.))             
        
            
    #计算并返回L2_mask
    stargan_prop_dist = float(n_dist) / n_samples
    return stargan_prop_dist


def evaluate_AttentionGAN(args_attack, test_dataloader, attentiongan_solver, up, method):
    
    #初始化
    n_dist, n_samples = 0, 0
    
    #遍历测试数据
    for idx, (img_a, att_a, c_org) in enumerate(test_dataloader):
        
        #加载图像和属性
        img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
        att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
        att_a = att_a.type(torch.float)
        
        #生成对抗样本和无攻击样本
        x_noattack_list, x_fake_list = attentiongan_solver.test_universal_model_level(idx, img_a, c_org, up, args_attack.AttentionGAN)
        
        #评估
        for j in range(len(x_fake_list)):
            gen_noattack = x_noattack_list[j]
            gen = x_fake_list[j]
            mask = abs(gen_noattack - img_a)
            mask = mask[0,0,:,:] + mask[0,1,:,:] + mask[0,2,:,:]
            mask[mask>0.5] = 1
            mask[mask<0.5] = 0
            if (((gen*mask - gen_noattack*mask)**2).sum() / (mask.sum()*3)) > 0.05:
                n_dist += 1
            n_samples += 1
            
        #保存图片
        #for j in range(len(x_fake_list)):
        #    # 保存原图生成图片
        #    gen_noattack = x_noattack_list[j]
        #    out_file = './demo_results/attentiongan_gen_{}.jpg'.format(j)
        #    vutils.save_image(gen_noattack, out_file, nrow=1, normalize=True, value_range=(-1., 1.))
        #    # 保存对抗样本生成图片
        #    gen = x_fake_list[j]
        #    out_file = './demo_results/attentiongan_advgen_{}.jpg'.format(j)
        #    vutils.save_image(gen, out_file, nrow=1, normalize=True, value_range=(-1., 1.))

    #计算并返回L2_mask
    aggan_prop_dist = float(n_dist) / n_samples
    return aggan_prop_dist

def evaluate_HIDF(args_attack, test_dataloader,transform, F, T, G, E, reference, gen_models, up, method):
    
    #初始化
    n_dist, n_samples = 0, 0
    
    #遍历测试数据
    for idx, (img_a, att_a, c_org) in enumerate(test_dataloader):
        img_a = img_a.cuda()  # 数据转移到GPU
        samples = [img_a, img_a+up]
        
        #生成对抗样本和无攻击样本
        with torch.no_grad():
            #无攻击样本生成
            #编码器提取特征
            c = E(img_a)
            #目标特征
            c_trg = c 
            # 风格提取
            s_trg = F(reference, 1)  
            # 特征转换
            c_trg = T(c_trg, s_trg, 1)  
            # 生成器产生干净样本
            gen_noattack = G(c_trg)  
            
            # 对抗样本生成
            # 添加扰动后的编码
            c = E(img_a + up)  
            c_trg = c
            s_trg = F(reference, 1)
            c_trg = T(c_trg, s_trg, 1)
            gen = G(c_trg)
            
            #评估
            mask = abs(gen_noattack - img_a).sum(1)
            mask = (mask > 0.5).float()
            if (((gen*mask - gen_noattack*mask)**2).sum() / (mask.sum()*3)) > 0.05:
                n_dist += 1
            n_samples += 1
            
            #保存图片
            #for j in range(len(samples)-2):
            #    # 保存对抗样本生成的图片
            #    out_file = './demo_results/HiSD_advgen_{}.jpg'.format(j)
            #    vutils.save_image(samples[j+2], out_file, nrow=1, normalize=True, value_range=(-1., 1.))
            #    # 保存原图生成的图片
            #    out_file = './demo_results/HiSD_gen_{}.jpg'.format(j)
            #    vutils.save_image(noattack_list[j], out_file, nrow=1, normalize=True, value_range=(-1., 1.))
            
    #计算并返回L2_mask
    HiDF_prop_dist = float(n_dist) / n_samples
    return HiDF_prop_dist

def evaluate_FGSM(args_attack, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models, up, method):
    stargan_prop_dist = evaluate_stargan(args_attack, test_dataloader,solver, up, method)
    attgan_prop_dist = evaluate_AttGAN(args_attack, test_dataloader, attgan, attgan_args, up, method)
    aggan_prop_dist = evaluate_AttentionGAN(args_attack, test_dataloader, attentiongan_solver, up, method)
    HiDF_prop_dist = evaluate_HIDF(args_attack, test_dataloader,transform, F, T, G, E, reference, gen_models, up, method)
    return HiDF_prop_dist, stargan_prop_dist, attgan_prop_dist, aggan_prop_dist