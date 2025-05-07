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
# from transformers import *
cmau_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(cmau_path)

from AttGAN.data import check_attribute_conflict
from data import CelebA
from model_data_prepare import prepare
from MIM.mim import MIMattack

def parse(args=None):
    """
    解析配置文件为参数对象
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(join(os.path.join(current_dir, 'setting.json')), 'r') as f:
        args_attack = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
    return args_attack

# def save_images(save_dir, image_list, name_list, idx):
#     """
#     保存生成的图像和扰动
    
#     参数:
#         save_dir: 保存目录
#         image_list: 图像列表
#         name_list: 图像名称列表
#         idx: 图像索引
#     """
#     # 确保目录存在
#     os.makedirs(save_dir, exist_ok=True)
    
#     for i, (image, name) in enumerate(zip(image_list, name_list)):
#         img_path = os.path.join(save_dir, f"{idx:06d}_{name}.jpg")
#         norm_image = (image + 1) / 2.0
#         vutils.save_image(
#             norm_image, img_path
#         )
    
#     print(f"保存图像 {idx} 完成")
def force_cudnn_initialization():
    """强制初始化cuDNN以避免首次运行时的峰值内存"""
    if torch.cuda.is_available():
        s = 32
        torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device='cuda'), 
                                    torch.zeros(s, s, s, s, device='cuda'))
        torch.cuda.empty_cache()

def save_images(save_dir, image_list, name_list, idx_start, batch_size=1):
    """
    保存生成的图像和扰动，支持批量处理
    
    参数:
        save_dir: 保存目录
        image_list: 图像列表
        name_list: 图像名称列表
        idx_start: 起始图像索引
        batch_size: 批量大小
    """
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    for i, (image, name) in enumerate(zip(image_list, name_list)):
        # 处理批量图像
        for b in range(image.size(0)):
            img_path = os.path.join(save_dir, f"{idx_start + b:06d}_{name}.jpg")
            norm_image = (image[b:b+1] + 1) / 2.0
            vutils.save_image(
                norm_image, img_path
            )
            del norm_image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    print(f"保存批次 {idx_start} 图像完成")

def main():
    """
    主函数，实现标准MIM攻击
    """

    # 加载配置
    args_attack = parse()
    print("配置加载完成")
    
    # 获取批量大小
    batch_size = args_attack.mim_attacks.batch_size
    print(f"批量大小: {batch_size}")
    
    # 创建结果目录
    # region ver2
    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_base_path = os.path.join(current_dir, 'MIM_results')
    os.makedirs(result_base_path, exist_ok=True)
    # endregion ver2


    # 拷贝配置文件到结果目录
    # region ver2
    setting_path = os.path.join(current_dir, 'setting.json')
    if os.path.exists(setting_path):
        os.system(f'cp "{setting_path}" "{os.path.join(result_base_path, "setting.json")}"')
        print(f"创建实验目录并复制配置文件: {setting_path} -> {os.path.join(result_base_path, 'setting.json')}")
    else:
        print(f"警告: 配置文件 {setting_path} 不存在")
    # endregion ver2
    
    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() and args_attack.global_settings.gpu else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        force_cudnn_initialization()

    # 加载模型
    print("初始化模型...")
    attack_dataloader, train_dataloader, attgan, attgan_args, stargan_solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = prepare(batch_size=batch_size)
    print("模型加载完成")
    
    # 创建结果子目录
    models_results_dirs = {
        'attgan': os.path.join(result_base_path, 'attgan'),
        'attentiongan': os.path.join(result_base_path, 'attentiongan'),
        'stargan': os.path.join(result_base_path, 'stargan'),
        'hisd': os.path.join(result_base_path, 'hisd')
    }
    
    for dir_path in models_results_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # 初始化MIM攻击器
    mim_attack = MIMattack(
        model=None, 
        device=device,
        epsilon=args_attack.mim_attacks.epsilon, 
        iterations=args_attack.mim_attacks.iterations,
        decay_factor=args_attack.mim_attacks.decay_factor,
    )
    
    print("开始进行标准MIM攻击测试...")
    
    # 创建保存扰动的字典
    perturbations = {}
    
    # 对每个模型进行攻击
    train_limit = train_limit = getattr(args_attack.mim_attacks, 'num_train', None) or len(train_dataloader)

    for batch_idx, (img_a, att_a, c_org) in enumerate(tqdm(train_dataloader)):
        if batch_idx * batch_size >= train_limit:
            break
        
        current_batch_size = img_a.size(0)
        
        # 将数据移动到设备上
        img_a = img_a.to(device)
        att_a = att_a.to(device)
        att_a = att_a.type(torch.float)
        
        # 1. 攻击AttGAN模型 ===================================================
        att_b_list = [att_a]
        attgan.to(device)
        attgan.eval()
        
        for i in range(attgan_args.n_attrs):
            tmp = att_a.clone()
            tmp[:, i] = 1 - tmp[:, i]
            tmp = check_attribute_conflict(tmp, attgan_args.attrs[i], attgan_args.attrs)
            att_b_list.append(tmp)
        
        # print("开始进行AttGAN攻击")
        # 对每个属性变换进行攻击
        attgan_images = [img_a]
        for i, att_b in enumerate(att_b_list):
            # 将属性编码转换为模型所需格式
            att_b_ = (att_b * 2 - 1) * attgan_args.thres_int
            if i > 0:
                att_b_[..., i - 1] = att_b_[..., i - 1] * args_attack.AttGAN.attgan_test_int/ attgan_args.thres_int
            
            with torch.no_grad():
                gen_noattack = attgan.G(img_a, att_b_)
            
            # AttGAN的MIM攻击
            x_adv, perturb = mim_attack.perturb_attgan(img_a, att_b_, gen_noattack, attgan.G)
            
            # 生成对抗样本
            with torch.no_grad():
                gen_adv = attgan.G(x_adv, att_b_)
            
            # 保存图像和扰动
            attgan_images.extend([x_adv, gen_adv])
            del gen_noattack, att_b_, x_adv, gen_adv
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 保存AttGAN的扰动
        perturbations['attgan'] = perturb.detach().cpu()
        
        # 保存AttGAN攻击结果
        # save_images(
        #     models_results_dirs['attgan'], 
        #     [img.detach().cpu() for img in attgan_images],
        #     ['original'] + [f'attr{i}_adv_input' for i in range(len(att_b_list) - 1)] + 
        #                 [f'attr{i}_adv_output' for i in range(len(att_b_list) - 1)],
        #     idx
        # )

        save_images(
            models_results_dirs['attgan'], 
            [img.detach().cpu() for img in attgan_images],
            ['original'] + [f'attr{i}_adv_input' for i in range(len(att_b_list) - 1)] + 
                        [f'attr{i}_adv_output' for i in range(len(att_b_list) - 1)],
            batch_idx * batch_size,
            current_batch_size
        )

        # print("结束AttGAN攻击")
        del attgan_images, img_a, att_a, att_b_list, perturb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        attgan.to('cpu')

        # 2. 攻击StarGAN模型 ==================================================
        # 获取目标域标签
        # print("开始进行StarGAN攻击")
        c_trg_list = stargan_solver.create_labels(c_org, stargan_solver.c_dim, stargan_solver.dataset, stargan_solver.selected_attrs)
        # stargan_solver.eval()
        stargan_images = [img_a]
        for i, c_trg in enumerate(c_trg_list):
            with torch.no_grad():
                gen_noattack, _ = stargan_solver.G(img_a, c_trg)
            
            # StarGAN的PGD攻击
            x_adv, perturb = mim_attack.perturb_stargan(img_a, gen_noattack, c_trg, stargan_solver.G)
            
            # 生成对抗样本输出
            with torch.no_grad():
                gen_adv, _ = stargan_solver.G(x_adv, c_trg)
            
            # 保存图像
            stargan_images.extend([x_adv, gen_adv])
        
        # 保存StarGAN的扰动
        perturbations['stargan'] = perturb.detach().cpu()
        
        # 保存StarGAN攻击结果
        # save_images(
        #     models_results_dirs['stargan'],
        #     [img.detach().cpu() for img in stargan_images],
        #     ['original'] + [f'domain{i}_adv_input' for i in range(len(c_trg_list))] + 
        #                 [f'domain{i}_adv_output' for i in range(len(c_trg_list))],
        #     idx
        # )

        save_images(
            models_results_dirs['stargan'], 
            [img.detach().cpu() for img in stargan_images],
            ['original'] + [f'domain{i}_adv_input' for i in range(len(c_trg_list))] + 
                        [f'domain{i}_adv_output' for i in range(len(c_trg_list))],
            batch_idx * batch_size,
            current_batch_size
        )

        # print("结束StarGAN攻击")
        del stargan_images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        
        # 3. 攻击AttentionGAN模型 ============================================
        # print("开始进行AttentionGAN攻击")
        attentiongan_images = [img_a]
        attentiongan_solver.G.eval()

        for i, c_trg in enumerate(c_trg_list):
            with torch.no_grad():
                gen_noattack, _, _  = attentiongan_solver.G(img_a, c_trg)
            
            # AttentionGAN的PGD攻击
            x_adv, perturb = mim_attack.perturb_attentiongan(img_a, gen_noattack, c_trg, attentiongan_solver.G)
            
            # 生成对抗样本输出
            with torch.no_grad():
                gen_adv, _, _ = attentiongan_solver.G(x_adv, c_trg)
            
            # 保存图像
            attentiongan_images.extend([x_adv, gen_adv])
        
        # 保存AttentionGAN的扰动
        perturbations['attentiongan'] = perturb.detach().cpu()
        
        # 保存AttentionGAN攻击结果
        # save_images(
        #     models_results_dirs['attentiongan'],
        #     [img.detach().cpu() for img in attentiongan_images],
        #     ['original'] + [f'domain{i}_adv_input' for i in range(len(c_trg_list))] + 
        #                 [f'domain{i}_adv_output' for i in range(len(c_trg_list))],
        #     idx
        # )

        save_images(
            models_results_dirs['attentiongan'], 
            [img.detach().cpu() for img in attentiongan_images],
            ['original'] + [f'domain{i}_adv_input' for i in range(len(c_trg_list))] + 
                        [f'domain{i}_adv_output' for i in range(len(c_trg_list))],
            batch_idx * batch_size,
            current_batch_size
        )

        # print("结束AttentionGAN攻击")
        # 释放内存
        del attentiongan_images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 4. 攻击HiSD模型 ====================================================
        # print("开始进行HiSD攻击")
        hisd_images = [img_a]
        batch_szize = img_a.size(0)

        if reference.size(0) != batch_szize:
            reference = reference.repeat(batch_szize, 1, 1, 1)
        else:
            reference = reference.clone().detach_().to(device)
        
        # 获取HiSD模型的输出
        with torch.no_grad():
            c = E(img_a)
            c_trg = c
            s_trg = F(reference, 1)
            c_trg = T(c_trg, s_trg, 1)
            x_trg = G(c_trg)
            
            # 计算掩码
            # mask = abs(x_trg - img_a)
            # mask = mask[0,0,:,:] + mask[0,1,:,:] + mask[0,2,:,:]
            # mask[mask>0.5] = 1
            # mask[mask<0.5] = 0
            # mask.to(device)
            masks = []
            for b in range(batch_size):
                mask = abs(x_trg[b:b+1] - img_a[b:b+1])
                mask = mask[0,0,:,:] + mask[0,1,:,:] + mask[0,2,:,:]
                mask[mask>0.5] = 1
                mask[mask<0.5] = 0
                masks.append(mask)
            mask = torch.stack(masks, dim=0).to(device)

        # HiSD的PGD攻击
        x_adv, perturb = mim_attack.perturb_hisd(img_a, reference, x_trg, E, F, T, G, gen_models, mask, transform)
        
        # 生成对抗样本输出
        with torch.no_grad():
            if transform is not None:
                x_adv_t = transform(x_adv)
            else:
                x_adv_t = x_adv
            
            c_adv = E(x_adv_t)
            # s_trg = F(reference, 1)
            if reference.size(0) != x_adv.size(0):
                reference_batch = reference.repeat(x_adv.size(0), 1, 1, 1)
                s_trg = F(reference_batch, 1)
            else:
                s_trg = F(reference, 1)
            c_trg_adv = T(c_adv, s_trg, 1)
            x_trg_adv = G(c_trg_adv)
        
        # 保存图像
        hisd_images.extend([x_adv, x_trg_adv])
        
        # 保存HiSD的扰动
        perturbations['hisd'] = perturb.detach().cpu()
        
        # 保存HiSD攻击结果
        # save_images(
        #     models_results_dirs['hisd'],
        #     [img.detach().cpu() for img in hisd_images],
        #     ['original', 'adv_input', 'adv_output'],
        #     idx
        # )
        save_images(
            models_results_dirs['hisd'], 
            [img.detach().cpu() for img in hisd_images],
            ['original', 'adv_input', 'adv_output'],
            batch_idx * batch_size,
            current_batch_size
        )
        # print("结束HiSD攻击")

        # 释放内存
        del hisd_images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 保存所有扰动
    for model_name, perturb in perturbations.items():
        torch.save(perturb, os.path.join(result_base_path, f'{model_name}_perturbation.pt'))
    
    print("标准MIM攻击测试完成，结果已保存至", result_base_path)
    print(f"扰动已保存至路径：{result_base_path}，文件名：model_name_perturbation.pt")
    print("所有模型的扰动已保存完成")
    

if __name__ == "__main__":
    main()