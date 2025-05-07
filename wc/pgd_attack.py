"""
用途：PGD水印生成
"""

import argparse
import json
import os
import torch
import gc
from tqdm import tqdm
from os.path import join

from StandardPGD.PGD import PGDAttack
from model_data_prepare import prepare
from AttGAN.data import check_attribute_conflict

def parse_args():
    """加载配置文件参数"""
    with open(join('./setting.json'), 'r') as f:
        args_attack = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
    return args_attack

def best_parse_args():
    """加载最佳参数"""
    with open(join('./best_params_pgd_mim.json'), 'r', encoding='utf-8') as f:
        best_params = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
    return best_params

def cleanup_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def main():
    # 加载配置参数
    args_attack= parse_args()
    best_params = best_parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and args_attack.global_settings.gpu else 'cpu')
    print(f"使用设备: {device}")

    #创建结果目录
        # 创建结果目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_path = current_dir
    os.makedirs(result_path, exist_ok=True)
    
    # 加载数据和模型
    print("初始化模型和数据...")
    attack_dataloader, _, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = prepare(batch_size=args_attack.pgd_attacks.batch_size)
    print("模型加载完成")
    
    # 初始化PGD攻击器
    pgd_attack = PGDAttack(
        device=device,
        epsilon=args_attack.standard_pgd_attacks.epsilon if hasattr(args_attack, 'standard_pgd_attacks') else 0.01,
        iterations=args_attack.standard_pgd_attacks.k if hasattr(args_attack, 'standard_pgd_attacks') else 10,
        step_size=args_attack.standard_pgd_attacks.a if hasattr(args_attack, 'standard_pgd_attacks') else 0.01
    )

    print("开始生成PGD扰动...")
    solver.G.eval()
    attentiongan_solver.G.eval()
    attgan.eval()
    cleanup_memory()

    # 使用PGD攻击
    print("总共的测试样本数:", args_attack.pgd_attacks.num_attack)
    print("批次大小:", args_attack.pgd_attacks.batch_size)
    print("循环轮数:",  args_attack.pgd_attacks.num_attack/args_attack.pgd_attacks.batch_size)

    
    perturb_stargan_list = []
    perturb_attention_list = []
    perturb_hisd_list = []
    perturb_attgan_list = []
    
    for idx,(img_a, att_a, c_org) in enumerate(tqdm(attack_dataloader, desc="PGD攻击进度")):


        if args_attack.pgd_attacks.num_attack is not None and idx * args_attack.pgd_attacks.batch_size >= args_attack.pgd_attacks.num_attack:
            break
        
        img_a = img_a.to(device)
        att_a = att_a.type(torch.float).to(device)

        # 1. 攻击StarGAN
        batch_star=[]
        pgd_attack.set_params(
            epsilon=best_params.pgd_stargan.epsilon,
            iterations=best_params.pgd_stargan.iterations,
            step_size=best_params.pgd_stargan.step_size,
        )

        solver.G.to(device)
        c_trg_list = solver.create_labels(c_org, solver.c_dim, solver.dataset, solver.selected_attrs)
        for c_trg in c_trg_list:
            with torch.no_grad():
                out_put, _ = solver.G(img_a, c_trg)
        
            _,perturb_stargan=pgd_attack.perturb_stargan(img_a,out_put,c_trg,solver.G)
            batch_star.append(perturb_stargan)
        perturb_stargan = torch.stack(batch_star, dim=0).mean(dim=0)
        perturb_stargan_list.append(perturb_stargan)

              
        del out_put, c_trg_list
        solver.G.to('cpu')
        cleanup_memory()

        # 2. 攻击AttentionGAN
        batch_attention = []
        attentiongan_solver.G.to(device)
        pgd_attack.set_params(
            epsilon=best_params.pgd_attentiongan.epsilon,
            iterations=best_params.pgd_attentiongan.iterations,
            step_size=best_params.pgd_attentiongan.step_size,
        )

        c_trg_list = attentiongan_solver.create_labels(
            c_org, attentiongan_solver.c_dim, attentiongan_solver.dataset, attentiongan_solver.selected_attrs
        )

        for c_trg in c_trg_list:
            with torch.no_grad():
                output, _, _ = attentiongan_solver.G(img_a, c_trg)
            _, perturb_attention = pgd_attack.perturb_attentiongan(img_a, output, c_trg, attentiongan_solver.G)
            batch_attention.append(perturb_attention)
        perturb_attention = torch.stack(batch_attention, dim=0).mean(dim=0)
        perturb_attention_list.append(perturb_attention)
        

        del c_trg_list
        attentiongan_solver.G.to('cpu')
        cleanup_memory()

        # 3. 攻击HiSD
        pgd_attack.set_params(
            epsilon=best_params.pgd_hisd.epsilon,
            iterations=best_params.pgd_hisd.iterations,
            step_size=best_params.pgd_hisd.step_size,
        )
        with torch.no_grad():
            c = E(img_a)
            c_trg = c
            s_trg = F(reference, 1)
            c_trg = T(c_trg, s_trg, 1)
            x_trg = G(c_trg)
            mask = abs(x_trg - img_a).sum(dim=1, keepdim=True)
            mask = (mask > 0.5).float()
        
        _, perturb_hisd = pgd_attack.perturb_hisd(
            img_a, reference, x_trg, E, F, T, G, gen_models, mask, None
        )
        perturb_hisd_list.append(perturb_hisd)

        cleanup_memory()

        # 4. 攻击AttGAN
        batch_attgan = []
        attgan.G.to(device)
        att_b_list = [att_a]
        pgd_attack.set_params(
            epsilon=best_params.pgd_attgan.epsilon,
            iterations=best_params.pgd_attgan.iterations,
            step_size=best_params.pgd_attgan.step_size,
        )

        for i in range(attgan_args.n_attrs):
            tmp= att_a.clone()
            tmp[:, i] = 1 - tmp[:, i]
            tmp = check_attribute_conflict(tmp, attgan_args.attrs[i], attgan_args.attrs)
            att_b_list.append(tmp)
        
        for i, att_b in enumerate(att_b_list):
            att_b_ = (att_b * 2 - 1) * attgan_args.thres_int
            if i > 0:
                att_b_[..., i - 1] = att_b_[..., i - 1] * attgan_args.test_int / attgan_args.thres_int

            with torch.no_grad():
                gen_noattack = attgan.G(img_a, att_b_)
            
            _, perturb_attgan = pgd_attack.perturb_attgan(img_a, att_b_, gen_noattack, attgan)
            batch_attgan.append(perturb_attgan)
        perturb_attgan = torch.stack(batch_attgan, dim=0).mean(dim=0)
        perturb_attgan_list.append(perturb_attgan)
        attgan.G.to('cpu')
        cleanup_memory()
        

    # 保存各模型的扰动
    perturb_stargan = torch.clamp(torch.stack(perturb_stargan_list, dim=0).mean(dim=0), min=-best_params.pgd_stargan.epsilon, max=+best_params.pgd_stargan.epsilon)
    perturb_attention = torch.clamp(torch.stack(perturb_attention_list, dim=0).mean(dim=0), min=-best_params.pgd_attentiongan.epsilon, max=+best_params.pgd_attentiongan.epsilon)
    perturb_hisd = torch.clamp(torch.stack(perturb_hisd_list, dim=0).mean(dim=0), min=-best_params.pgd_hisd.epsilon, max=+best_params.pgd_hisd.epsilon)
    perturb_attgan = torch.clamp(torch.stack(perturb_attgan_list, dim=0).mean(dim=0), min=-best_params.pgd_attgan.epsilon, max=+best_params.pgd_attgan.epsilon)
    torch.save(perturb_stargan, os.path.join(result_path, 'pgd_stargan_perturbation.pt'))
    torch.save(perturb_attention, os.path.join(result_path, 'pgd_attentiongan_perturbation.pt'))
    torch.save(perturb_hisd, os.path.join(result_path, 'pgd_hisd_perturbation.pt'))
    torch.save(perturb_attgan, os.path.join(result_path, 'pgd_attgan_perturbation.pt'))

    print(f"扰动已保存至: {result_path}")

if __name__ == "__main__":
    main()
    cleanup_memory()
