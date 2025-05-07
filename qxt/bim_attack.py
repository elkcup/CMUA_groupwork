# bim_attack.py
import copy
import numpy as np
from scipy.stats import truncnorm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.utils as vutils

try:
    import defenses.smoothing as smoothing
except:
    import AttGAN.defenses.smoothing as smoothing

class BIMAttack(object):
    def __init__(self, model=None, device=None, epsilon=0.05, k=10, a=None, star_factor=0.3, attention_factor=0.3, att_factor=2, HiSD_factor=1, feat=None, args=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model  # 目标模型
        self.epsilon = epsilon  # 最大扰动
        self. k = k  # 迭代次数
        self.a = a if a is not None else epsilon / k  # BIM步长
        self.loss_fn = nn.MSELoss().to(device)  # 损失函数
        self.device = device  # 计算设备
        self.feat = feat
        
        self.rand = False  # 不对输入样本添加随机初始化的扰动，直接从原始输入样本开始攻击。 
        
        # 通用扰动
        self.up = None  
        self.att_up = None 
        self.attention_up = None  
        self.star_up = None  
        self.HISD_up=None
        self.momentum = args.momentum
        
        self.window_size = 3  # 统一控制窗口大小
        self.stargan_grad_window = []
        self.attentiongan_grad_window = [] 
        self.attgan_grad_window = []
        self.hisd_grad_window = []
        
        # 模型权重参数
        self.star_factor = star_factor  
        self.attention_factor = attention_factor
        self.att_factor = att_factor
        self.HiSD_factor = HiSD_factor
    
    # BIM核心攻击方法
    def perturb(self, X_nat, y, model=None, c_trg=None):  # X_nat原始样本（图像）, y目标输出, c_trg目标域标签
        X = X_nat.clone().detach_()  # 直接从原始样本开始
        """
        clone()创建一个与原张量具有相同数据和形状的新张量，但它们在内存中是独立的
        detach_()将张量从当前计算图中分离出来，使其不再需要梯度计算
        """
        for i in range(self.k):
            X.requires_grad = True  # 将 X 设置为可求导，以便计算梯度
            output, feats = self.model(X, c_trg)  # 向前传播

            if self.feat:
                output = feats[self.feat]  # 如果指定了某个中间层特征，使用该特征作为攻击目标。

            self.model.zero_grad()  # 清空梯度，避免梯度累积            
            loss = self.loss_fn(output, y)  # 计算模型输出与目标标签之间的损失  
            loss.backward()  # 反向传播，计算梯度
            
            grad = X.grad  # 获取 X 的梯度            
            X_adv = X + self.a * grad.sign()  
            
            # 确保扰动在epsilon范围内
            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        self.model.zero_grad()
        return X, X - X_nat  # 返回最终的对抗样本 X 和扰动 X - X_nat

    # 对AttGAN模型的攻击
    def universal_perturb_attgan(self, X_nat, X_att, y, attgan):  # X_nat原始样本， X_att属性向量，指定要改变的人脸属性，y目标输出，attgan包含生成器 G 的 GAN 模型
            # 初始化通用扰动和历史记录
        if self.up is None:
            self.up = torch.zeros_like(X_nat[0]).unsqueeze(0).to(self.device)  # 保持维度[1,C,H,W]
            self.attgan_grad_window.clear()  # 新增梯度窗口缓存
        
        X = X_nat.clone().detach_()

        for i in range(self.k):
            X.requires_grad = True
            output = attgan.G(X, X_att)

            attgan.G.zero_grad()
            loss = self.loss_fn(output, y)
            loss.backward()
            
            grad = X.grad.detach()  # 显式分离梯度
            
            # 计算当前步扰动
            current_step = self.att_factor * self.a * grad.sign()
            batch_avg_step = torch.mean(current_step, dim=0, keepdim=True)  # 批次平均 

            # 更新梯度窗口（保存最近3步）
            self.attgan_grad_window.append(batch_avg_step)
            if len(self.attgan_grad_window) > 3:
                self.attgan_grad_window.pop(0)

            # 计算窗口平均
            if len(self.attgan_grad_window) > 0:
                avg_step = torch.stack(self.attgan_grad_window).mean(dim=0)
            else:
                avg_step = batch_avg_step
        
            # 更新通用扰动（带约束）
            self.up += avg_step
            self.up = torch.clamp(self.up, -self.epsilon, self.epsilon)
        
            # 更新当前样本
            X = torch.clamp(X_nat + self.up.expand_as(X_nat), min=-1, max=1).detach_()

        attgan.G.zero_grad()
        return X, self.up.expand_as(X_nat) 
            
     
    
     # 对StarGAN模型的攻击
    def universal_perturb_stargan(self, X_nat, y, c_trg, model):
        if self.up is None:
            self.up = torch.zeros_like(X_nat[0]).unsqueeze(0).to(self.device)
            self.stargan_grad_window.clear()

        X = X_nat.clone().detach_()

        for i in range(self.k):
            X.requires_grad = True
            output, feats = model(X, c_trg)
            if self.feat:
                output = feats[self.feat]

            model.zero_grad()
            loss = self.loss_fn(output, y)
            loss.backward()
            grad = X.grad.detach()

            # 计算当前步扰动
            current_step = self.star_factor * self.a * grad.sign()
            batch_avg_step = torch.mean(current_step, dim=0, keepdim=True)

            # 更新梯度窗口（保存最近3步）
            self.stargan_grad_window.append(batch_avg_step)
            if len(self.stargan_grad_window) > 3:
                self.stargan_grad_window.pop(0)

            # 计算窗口平均
            avg_step = torch.stack(self.stargan_grad_window).mean(dim=0) if self.stargan_grad_window else batch_avg_step

            # 更新通用扰动
            self.up += avg_step
            self.up = torch.clamp(self.up, -self.epsilon, self.epsilon)

            X = torch.clamp(X_nat + self.up.expand_as(X_nat), min=-1, max=1).detach_()

        model.zero_grad()
        return X, self.up.expand_as(X_nat)


    # 对AttentionGAN模型的攻击
    def universal_perturb_attentiongan(self, X_nat, y, c_trg, model):
        # 初始化通用扰动和梯度窗口
        if self.up is None:
            self.up = torch.zeros_like(X_nat[0]).unsqueeze(0).to(self.device)
            self.attentiongan_grad_window.clear()

        X = X_nat.clone().detach_()

        for i in range(self.k):
            X.requires_grad = True
            output, _, _ = model(X, c_trg)
            if self.feat:
                output = feats[self.feat]

            model.zero_grad()
            loss = self.loss_fn(output, y)
            loss.backward()
            grad = X.grad.detach()

            # 计算当前步扰动
            current_step = self.attention_factor * self.a * grad.sign()
            batch_avg_step = torch.mean(current_step, dim=0, keepdim=True)

            # 更新梯度窗口
            self.attentiongan_grad_window.append(batch_avg_step)
            if len(self.attentiongan_grad_window) > 3:
                self.attentiongan_grad_window.pop(0)

            # 计算窗口平均
            avg_step = torch.stack(self.attentiongan_grad_window).mean(dim=0) if self.attentiongan_grad_window else batch_avg_step

            # 更新通用扰动
            self.up += avg_step
            self.up = torch.clamp(self.up, -self.epsilon, self.epsilon)

            X = torch.clamp(X_nat + self.up.expand_as(X_nat), min=-1, max=1).detach_()

        model.zero_grad()
        return X, self.up.expand_as(X_nat)

   # 对HiSD模型的攻击
    def universal_perturb_HiSD(self, X_nat, transform, F, T, G, E, reference, y, gen, mask):
        # 初始化通用扰动和梯度窗口
        if self.up is None:
            self.up = torch.zeros_like(X_nat[0]).unsqueeze(0).to(self.device)
            self.hisd_grad_window.clear()

        X = X_nat.clone().detach_()

        for i in range(self.k):
            X.requires_grad = True
            c = E(X)
            c_trg = c
            s_trg = F(reference, 1)
            c_trg = T(c_trg, s_trg, 1)
            x_trg = G(c_trg)

            gen.zero_grad()
            loss = self.loss_fn(x_trg, y)
            loss.backward()
            grad = X.grad.detach()

            # 计算当前步扰动
            current_step = self.HiSD_factor * self.a * grad.sign()
            batch_avg_step = torch.mean(current_step, dim=0, keepdim=True)
    
            # 更新梯度窗口
            self.hisd_grad_window.append(batch_avg_step)
            if len(self.hisd_grad_window) > 3:
                self.hisd_grad_window.pop(0)

            # 计算窗口平均
            avg_step = torch.stack(self.hisd_grad_window).mean(dim=0) if self.hisd_grad_window else batch_avg_step

            # 更新通用扰动
            self.up += avg_step
            self.up = torch.clamp(self.up, -self.epsilon, self.epsilon)

            X = torch.clamp(X_nat + self.up.expand_as(X_nat), min=-1, max=1).detach_()

        gen.zero_grad()
        return X, self.up.expand_as(X_nat)
        
        
def clip_tensor(X, Y, Z):# 将输入张量 X 的值限制在两个张量 Y 和 Z 的范围内
    X_np = X.data.cpu().numpy()
    Y_np = Y.data.cpu().numpy()
    Z_np = Z.data.cpu().numpy()
    X_clipped = np.clip(X_np, Y_np, Z_np)
    X_res = torch.FloatTensor(X_clipped)
    return X_res

def perturb_batch(X, y, c_trg, model, adversary): # 用于对抗训练的扰动批次函数   
    model_cp = copy.deepcopy(model)  # 创建模型副本
    for p in model_cp.parameters():
        p.requires_grad = False  # 冻结模型参数
    model_cp.eval()  # 设置模型为评估模式
    
    adversary.model = model_cp # 将复制模型传递给对抗攻击工具，确保攻击工具使用的是冻结参数的模型。

    X_adv, _ = adversary.perturb(X, y, c_trg)  # 生成对抗样本

    return X_adv
