import argparse
import json
import os
import torch
import torch.utils.data as data
from model_data_prepare import prepare

import sys
from contextlib import contextmanager
import logging


from StandardPGD.PGD import PGDAttack
from MIM.mim import MIMattack
from evaluate_pgd_mim import evaluate_pgd, evaluate_mim
from evaluate_cmua_apgd import evaluate_apgd,evaluate_cmua
from auto_fgsm_evaluate import evaluate_FGSM
from bim_evaluate import evaluate_bim

import matplotlib.pyplot as plt
import numpy as np

# 屏蔽部分输出
@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull

        logging.basicConfig(level=logging.ERROR)
        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        old_levels = {}
        for logger in loggers:
            old_levels[logger.name] = logger.level
            logger.setLevel(logging.ERROR)
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            for logger in loggers:
                if logger.name in old_levels:
                    logger.setLevel(old_levels[logger.name])

# 加载配置
def parse_config():
    """解析配置文件为参数对象"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'setting.json')
    with open(config_path, 'r') as f:
        args_attack = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
    return args_attack


# 初始化
args_attack = parse_config()
device = torch.device('cuda' if torch.cuda.is_available() and args_attack.global_settings.gpu else 'cpu')
# os.system('cls' if os.name == 'nt' else 'clear')

# 加载模型和数据
print("加载模型和数据...")
with suppress_output():
    attack_dataloader, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = prepare()

print("模型和数据加载完成！")


subset = data.Subset(test_dataloader.dataset, list(range(129,134)))
sub_loader = data.DataLoader(subset, batch_size=1, shuffle=False)

current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, 'outputs')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
#################################
#           PGD和MIM评估        # @Wang Chao
#################################
# 评估PGD攻击
pgd_attack = PGDAttack()
pgd_hisd_dist, pgd_attgan_dist, pgd_attentiongan_dist, pgd_stargan_dist = evaluate_pgd(
    args_attack, sub_loader, attgan, attgan_args, solver, 
    attentiongan_solver, transform, F, T, G, E, reference)
pgd_results = {
    "attgan": pgd_attgan_dist,
    "stargan": pgd_stargan_dist,
    "attentiongan": pgd_attentiongan_dist,
    "hisd": pgd_hisd_dist
}

# 评估MIM攻击
mim_attack = MIMattack()
mim_hisd_dist, mim_attgan_dist, mim_attentiongan_dist, mim_stargan_dist = evaluate_mim(
    args_attack, sub_loader, attgan, attgan_args, solver, 
    attentiongan_solver, transform, F, T, G, E, reference)
mim_results = {
    "attgan": mim_attgan_dist,
    "stargan": mim_stargan_dist,
   "attentiongan": mim_attentiongan_dist,
    "hisd": mim_hisd_dist
}



##################################
##       CMAU和AutoPGD评估       # 需要有auto_pgd_perturbation.pt文件 @Liu Yifei
##################################
# 评估AutoPGD

apgd_hisd_dist,apgd_attgan_dist,apgd_attentiongan_dist, apgd_stargan_dist = evaluate_apgd(args_attack, sub_loader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models)

# 评估CMAU
cmua_hisd_dist,cmua_attgan_dist,cmua_attentiongan_dist, cmua_stargan_dist = evaluate_cmua(args_attack, sub_loader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models)




#################################
#       DI2-FGSM和M-DI2-FGSM评估       # 当前目录需要有DI2_FGSM_perturbation.pt、M_DI2_FGSM_perturbation.pt文件          @Huang Yixuan
#################################

DI2_up=torch.load('DI2_FGSM_perturbation.pt')
M_DI2_up=torch.load('M_DI2_FGSM_perturbation.pt')
#评估DI2-FGSM攻击
D_fgsm_HiDF_sr, D_fgsm_stargan_sr, D_fgsm_AttGAN_sr, D_fgsm_AttentionGAN_sr=evaluate_FGSM(args_attack, sub_loader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models, DI2_up, 'DI2_FGSM')
#评估M-DI2-FGSM攻击
MD_fgsm_HiDF_sr, MD_fgsm_stargan_sr, MD_fgsm_AttGAN_sr, MD_fgsm_AttentionGAN_sr=evaluate_FGSM(args_attack, sub_loader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models, M_DI2_up, 'M_DI2_FGSM')



##################################
##       BIM评估                 # 需要有bim_perturbation.pt文件 @Qi Xueting
##################################

# 评估BIM
bim_hisd_dist,bim_attgan_dist,bim_attentiongan_dist, bim_stargan_dist = evaluate_bim(args_attack, sub_loader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models)




## 结果输出
###### FGSM AND HIDF       @Huang Yixuan
D_fgsm_avg = (D_fgsm_HiDF_sr + D_fgsm_stargan_sr + D_fgsm_AttGAN_sr + D_fgsm_AttentionGAN_sr) / 4
MD_avg = (MD_fgsm_HiDF_sr + MD_fgsm_stargan_sr + MD_fgsm_AttGAN_sr + MD_fgsm_AttentionGAN_sr) / 4


# ##### PGD AND MIN      @Wang Chao
# # 计算平均成功率
pgd_avg = (pgd_hisd_dist + pgd_stargan_dist + pgd_attgan_dist + pgd_attentiongan_dist) / 4
mim_avg = (mim_hisd_dist + mim_stargan_dist + mim_attgan_dist + mim_attentiongan_dist) / 4
# print(f"平均成功率\t{pgd_avg:.4f}\t\t{mim_avg:.4f}")


####### CMUA AND APGD       @Liu Yifei
# 计算平均成功率
cmua_avg = (cmua_hisd_dist + cmua_stargan_dist + cmua_attgan_dist + cmua_attentiongan_dist) / 4
apgd_avg = (apgd_hisd_dist + apgd_stargan_dist + apgd_attgan_dist + apgd_attentiongan_dist) / 4


####### BIM       @Qi Xueting
# 计算平均成功率
bim_avg = (bim_hisd_dist + bim_stargan_dist + bim_attgan_dist + bim_attentiongan_dist) / 4


##为输出成功，假定的值  后面删除这几行代码

# pgd_hisd_dist, pgd_attgan_dist, pgd_attentiongan_dist, pgd_stargan_dist=1,1,1,1

# mim_hisd_dist, mim_attgan_dist, mim_attentiongan_dist, mim_stargan_dist=1,1,1,1
# pgd_avg=1
# mim_avg=1

print("\n========成功率============")
print("模型\t\tCMUA成功率\tAutoPGD成功率\tDI2-FGSM成功率\tM-DI2-FGSM成功率\tPGD成功率\tMIM成功率\tBIM成功率")
print(f"HiSD\t\t{cmua_hisd_dist:.4f}\t\t{apgd_hisd_dist:.4f}\t\t{D_fgsm_HiDF_sr:.4f}\t\t{MD_fgsm_HiDF_sr:.4f}\t\t\t{pgd_hisd_dist:.4f}\t\t{mim_hisd_dist:.4f}\t\t{bim_hisd_dist:.4f}")
print(f"StarGAN\t\t{cmua_stargan_dist:.4f}\t\t{apgd_stargan_dist:.4f}\t\t{D_fgsm_stargan_sr:.4f}\t\t{MD_fgsm_stargan_sr:.4f}\t\t\t{pgd_stargan_dist:.4f}\t\t{mim_stargan_dist:.4f}\t\t{bim_stargan_dist:.4f}")
print(f"AttGAN\t\t{cmua_attgan_dist:.4f}\t\t{apgd_attgan_dist:.4f}\t\t{D_fgsm_AttGAN_sr:.4f}\t\t{MD_fgsm_AttGAN_sr:.4f}\t\t\t{pgd_attgan_dist:.4f}\t\t{mim_attgan_dist:.4f}\t\t{bim_attgan_dist:.4f}")
print(f"AttentionGAN\t{cmua_attentiongan_dist:.4f}\t\t{apgd_attentiongan_dist:.4f}\t\t{D_fgsm_AttentionGAN_sr:.4f}\t\t{MD_fgsm_AttentionGAN_sr:.4f}\t\t\t{pgd_attentiongan_dist:.4f}\t\t{mim_attentiongan_dist:.4f}\t\t{bim_attentiongan_dist:.4f}")


print(f"平均成功率\t{cmua_avg:.4f}\t\t{apgd_avg:.4f}\t\t{D_fgsm_avg:.4f}\t\t{MD_avg:.4f}\t\t\t{pgd_avg:.4f}\t\t{mim_avg:.4f}\t\t{bim_avg:.4f}")




#################################
#       绘制图片
#################################


# 模型名称
models = ['AttGAN', 'stargan', 'AttentionGAN', 'HiSD']
num_models = len(models)

# 每个模型对应的7个变量
data = {
    'AttGAN': [
        pgd_attgan_dist,       
        mim_attgan_dist,       
        cmua_attgan_dist,      
        apgd_attgan_dist,      
        D_fgsm_AttGAN_sr,      
        MD_fgsm_AttGAN_sr,
        bim_attgan_dist
    ],
    'stargan': [
        pgd_stargan_dist,
        mim_stargan_dist,
        cmua_stargan_dist,
        apgd_stargan_dist,
        D_fgsm_stargan_sr,
        MD_fgsm_stargan_sr,
        bim_stargan_dist
    ],
    'AttentionGAN': [
        pgd_attentiongan_dist,
        mim_attentiongan_dist,
        cmua_attentiongan_dist,
        apgd_attentiongan_dist,
        D_fgsm_AttentionGAN_sr,
        MD_fgsm_AttentionGAN_sr,
        bim_attentiongan_dist
    ],
    'HiSD': [
        pgd_hisd_dist,
        mim_hisd_dist,
        cmua_hisd_dist,
        apgd_hisd_dist,
        D_fgsm_HiDF_sr,
        MD_fgsm_HiDF_sr,
        bim_hisd_dist
    ]
}

# 将数据转换为4x6的二维数组（行：模型，列：变量）
data_array = np.array([data[model] for model in models])

# 设置参数
spacing = 2.5  # 模型组之间的间隔
width = 0.3     # 每个条形的宽度
num_bars_per_model = 7

# 生成中心位置（使用更大的间隔）
x = np.arange(num_models) * spacing  # 模型组的中心位置

# 创建图形
fig, ax = plt.subplots(figsize=(16, 6))  # 加宽图形

# 定义颜色和标签
colors = plt.cm.tab20.colors[:num_bars_per_model]  # 使用10种颜色的前6种
labels = [
    'PGD', 'MIM', 'CMUA', 'AutoPGD', 'D-FGSM', 'MD-FGSM', 'BIM'
]  # 根据变量名提取攻击方法作为标签

# 绘制每个条形
for i in range(num_bars_per_model):
    # 计算每个条形的x位置（围绕模型中心对称）
    offset = (i - (num_bars_per_model - 1)/2) * width * 1.1 
    ax.bar(
        x + offset,
        data_array[:, i],  # 当前列的数据
        width=width,
        color=colors[i],
        label=labels[i] if i < len(labels) else f'Category {i+1}'
    )

# 设置坐标轴标签和刻度
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel('L2_mask')
ax.set_title('L2_mask Comparison Across Models')

# 添加图例和调整布局
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncols=3)
plt.tight_layout(rect=[0, 0, 0.85, 1])  # 为图例保留空间

# 显示图形
plt.savefig('SR.png')
