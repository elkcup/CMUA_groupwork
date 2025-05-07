"""
用途：MIM水印生成
"""

import argparse
import json
import os
import torch
import gc
from tqdm import tqdm
from os.path import join

from MIM.mim import MIMattack
from model_data_prepare import prepare

def parse_args():
    """加载配置文件参数"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(current_dir, "setting.json")
    with open(result_path, 'r') as f:
        args_attack = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
    return args_attack

def best_parse_args():
    """加载最佳参数"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(current_dir, "best_params_pgd_mim.json")
    with open(result_path, 'r', encoding='utf-8') as f:
        best_params = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
    return best_params

def cleanup_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def main():
    # 加载配置参数
    args_attack = parse_args()
    best_params = best_parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and args_attack.global_settings.gpu else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建结果目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_path = current_dir
    os.makedirs(result_path, exist_ok=True)
    
    # 加载数据和模型
    print("初始化模型和数据...")
    attack_dataloader, _, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = prepare(batch_size=args_attack.mim_attacks.batch_size)
    best_params = best_parse_args()
    print("模型加载完成")
    
    # 初始化MIM攻击器
    mim_attack = MIMattack(
        device=device,
        epsilon=args_attack.mim_attacks.epsilon if hasattr(args_attack, 'mim_attacks') else 0.01,
        iterations=args_attack.mim_attacks.iterations if hasattr(args_attack, 'mim_attacks') else 10,
        decay_factor=args_attack.mim_attacks.decay_factor if hasattr(args_attack, 'mim_attacks') else 0.5
    )
    
    print("开始生成MIM扰动...")
    solver.G.eval()
    attentiongan_solver.G.eval()
    attgan.eval()
    cleanup_memory()

    # 使用MIM攻击
    print("总共的测试样本数:", args_attack.mim_attacks.num_attack)
    print("批次大小:", args_attack.mim_attacks.batch_size)
    print("循环轮数:",  args_attack.mim_attacks.num_attack/args_attack.mim_attacks.batch_size)

    perturb_stargan_list=[]
    perturb_attention_list=[]
    perturb_hisd_list=[]
    perturb_attgan_list=[]

    
    for idx, (img_a, att_a, c_org) in enumerate(tqdm(attack_dataloader, desc="MIM攻击进度")):
        if args_attack.mim_attacks.num_attack is not None and idx * args_attack.mim_attacks.batch_size >= args_attack.mim_attacks.num_attack:
            break
        
        img_a = img_a.to(device)
        att_a = att_a.type(torch.float).to(device)
        
        # 1. 攻击StarGAN
        batch_star=[]
        mim_attack.set_params(
            epsilon=best_params.mim_stargan.epsilon,
            iterations=best_params.mim_stargan.iterations,
            decay_factor=best_params.mim_stargan.decay_factor,
            )
        solver.G.to(device)
        c_trg_list = solver.create_labels(c_org, solver.c_dim, solver.dataset, solver.selected_attrs)
        for c_trg in c_trg_list:
            with torch.no_grad():
                output, _ = solver.G(img_a, c_trg)
            _, perturb_stargan = mim_attack.perturb_stargan(img_a, output, c_trg, solver.G)
            batch_star.append(perturb_stargan)
        perturb_stargan = torch.stack(batch_star, dim=0).mean(dim=0)    
        perturb_stargan_list.append(perturb_stargan)
        
            
        del c_trg_list
        solver.G.to('cpu')
        cleanup_memory()
        
        # 2. 攻击AttentionGAN
        batch_attention = []
        mim_attack.set_params(
            epsilon=best_params.mim_attentiongan.epsilon,
            iterations=best_params.mim_attentiongan.iterations,
            decay_factor=best_params.mim_attentiongan.decay_factor,
            )

        attentiongan_solver.G.to(device)
        c_trg_list = attentiongan_solver.create_labels(
            c_org, attentiongan_solver.c_dim, attentiongan_solver.dataset, attentiongan_solver.selected_attrs
        )
        
        for c_trg in c_trg_list:
            with torch.no_grad():
                output, _, _ = attentiongan_solver.G(img_a, c_trg)
            _, perturb_attention = mim_attack.perturb_attentiongan(img_a, output, c_trg, attentiongan_solver.G)
            batch_attention.append(perturb_attention)
        perturb_attention = torch.stack(batch_attention, dim=0).mean(dim=0)
        perturb_attention_list.append(perturb_attention)
        
        
        del c_trg_list
        attentiongan_solver.G.to('cpu')
        cleanup_memory()

        # 3. 攻击HiSD

        mim_attack.set_params(
            epsilon=best_params.mim_hisd.epsilon,
            iterations=best_params.mim_hisd.iterations,
            decay_factor=best_params.mim_hisd.decay_factor,
            )


        with torch.no_grad():
            c = E(img_a)
            c_trg = c
            s_trg = F(reference, 1)
            c_trg = T(c_trg, s_trg, 1)
            x_trg = G(c_trg)
            mask = abs(x_trg - img_a).sum(dim=1, keepdim=True)
            mask = (mask > 0.5).float()
        
        _, perturb_hisd = mim_attack.perturb_hisd(
            img_a, reference, x_trg, E, F, T, G, gen_models, mask, None
        )
        perturb_hisd_list.append(perturb_hisd)
        
        # 4. 攻击AttGAN
        batch_attgan = []
        mim_attack.set_params(
            epsilon=best_params.mim_attgan.epsilon,
            iterations=best_params.mim_attgan.iterations,
            decay_factor=best_params.mim_attgan.decay_factor,
            )
        from AttGAN.data import check_attribute_conflict
        attgan.G.to(device)

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
            
            _, perturb_attgan = mim_attack.perturb_attgan(img_a, att_b_, gen_noattack, attgan)
            batch_attgan.append(perturb_attgan)
        perturb_attgan = torch.stack(batch_attgan, dim=0).mean(dim=0)
        perturb_attgan_list.append(perturb_attgan)

        
        attgan.G.to('cpu')
        cleanup_memory()
        
    # 保存各模型的扰动   
    perturb_stargan = torch.clamp(torch.stack(perturb_stargan_list, dim=0).mean(dim=0), min=-best_params.mim_stargan.epsilon, max=+best_params.mim_stargan.epsilon)
    perturb_attention = torch.clamp(torch.stack(perturb_attention_list, dim=0).mean(dim=0), min=-best_params.mim_attentiongan.epsilon, max=+best_params.mim_attentiongan.epsilon)
    perturb_hisd = torch.clamp(torch.stack(perturb_hisd_list, dim=0).mean(dim=0), min=-best_params.mim_hisd.epsilon, max=+best_params.mim_hisd.epsilon)
    perturb_attgan = torch.clamp(torch.stack(perturb_attgan_list, dim=0).mean(dim=0), min=-best_params.mim_attgan.epsilon, max=+best_params.mim_attgan.epsilon)
    torch.save(perturb_stargan, os.path.join(result_path, 'mim_stargan_perturbation.pt'))
    torch.save(perturb_attention, os.path.join(result_path, 'mim_attentiongan_perturbation.pt'))
    torch.save(perturb_hisd, os.path.join(result_path, 'mim_hisd_perturbation.pt'))
    torch.save(perturb_attgan, os.path.join(result_path, 'mim_attgan_perturbation.pt'))
    print(f"扰动已保存至: {result_path}")

if __name__ == "__main__":
    main()
    cleanup_memory()

    print("MIM攻击完成，扰动已保存")