import copy
import numpy as np
from collections.abc import Iterable
from scipy.stats import truncnorm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.utils as vutils

try:
    import defenses.smoothing as smoothing
except:
    import AttGAN.defenses.smoothing as smoothing

class FGSMAttack(object):
    #初始化参数
    def __init__(self, model=None, device=10, epsilon=16.0, k=10, a=2.0 / 255.0 * 2, momentum = 0.9,
                 star_factor=0.3, attention_factor=0.3, att_factor=2, 
                 HiSD_factor=1, feat=None, args=None, prob=0.1, mode='universal', method='DI2_FGSM'):
        #目标模型
        self.model = model
        #扰动最大幅度
        self.epsilon = epsilon / 255.0 * 2 
        #迭代次数
        self.k = int(min(epsilon + 4, 1.25 * epsilon)) if k is None else k  
        #通用攻击步长
        self.a = a 
        #输入多样性应用概率(p)
        self.prob = prob  
        #损失函数
        self.loss_fn = nn.MSELoss().to(device)
        #设备
        self.device = device
        #通用扰动
        self.up = None
        #AttGAN专用扰动
        self.att_up = None
        #AttentionGAN专用扰动
        self.attention_up = None
        #stargan专用扰动
        self.star_up = None
        #HiSD专用扰动
        self.HISD_up=None
        #动量系数(μ)，为0表示基于DI2-FGSM攻击训练，不为0表示基于M-DI2-FGSM攻击训练
        self.momentum = momentum if method=='M_DI2_FGSM' else 0
        #AttGAN专用攻击步长
        self.att_factor = att_factor
        #AttentionGAN专用攻击步长
        self.attention_factor = attention_factor
        #stargan专用攻击步长
        self.star_factor = star_factor
        #HiSD专用攻击步长
        self.HiSD_factor = HiSD_factor
        #攻击模式（通用/专用），可以取值'universal','attgan','stargan','attention','HiSD'
        self.mode = mode
        #使用方法
        self.method=method#可以取值['DI2_FGSM'，'M_DI2_FGSM']
    #创建多样性输入
    def input_diversity(self, x):#x.shape=[3,256,256]
        
        # 以prob概率应用随机缩放和填充
        if np.random.rand() > self.prob:
            return x
        
        # 生成随机缩放尺寸（原图尺寸的90%-100%）
        min_size = int(x.shape[2] * 0.9)
        max_size = x.shape[2]
        new_size = np.random.randint(min_size, max_size + 1) if min_size < max_size else max_size
        
        #计算高度和宽度方向填充量
        h_rem = x.shape[2] - new_size
        w_rem = x.shape[2] - new_size
        
        #随机生成顶部填充量
        pad_top = np.random.randint(0, h_rem + 1) if h_rem > 0 else 0
        #计算底部填充量
        pad_bottom = h_rem - pad_top
        #随机生成左侧填充量
        pad_left = np.random.randint(0, w_rem + 1) if w_rem > 0 else 0
        #计算右侧填充量
        pad_right = w_rem - pad_left
        
        #使用最近邻插值（与DI2-FGSM原始代码保持一致）将图像放缩到new_size*new_size
        x_resized = torch.nn.functional.interpolate(x,size=(new_size, new_size),mode='nearest')
        
        # 填充操作，填充值为0（黑色）
        x_padded = torch.nn.functional.pad(
            x_resized,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode='constant',
            value=0
        )
        return x_padded
    
    def universal_perturb_attgan(self, X_nat, X_att, y, attgan):
        
        #复制原始数据并分离
        X = X_nat.clone().detach_()
        
        #进行迭代
        for i in range(self.k):
            X.requires_grad = True
            
            #应用多样性输入
            X_diverse=self.input_diversity(X)
            
            #生成输出
            output = attgan.G(X_diverse, X_att)
            attgan.G.zero_grad()
            
            #计算损失和梯度
            loss = self.loss_fn(output, y)
            loss.backward()
            grad = X.grad
            
            #更新对抗样本和扰动
            X_adv = X + self.a * grad.sign()
            if self.mode=='universel':
                if self.up is None:
                    eta = torch.mean(torch.clamp(self.att_factor*(X_adv - X_nat), min=-self.epsilon, max=self.epsilon).detach_(), dim=0)
                    self.up = eta
                else:
                    eta = torch.mean(torch.clamp(self.att_factor*(X_adv - X_nat), min=-self.epsilon, max=self.epsilon).detach_(), dim=0)
                    self.up = self.up * self.momentum + eta * (1 - self.momentum)
                X = torch.clamp(X_nat + self.up, min=-1, max=1).detach_()
            elif self.mode=='attgan':
                if self.att_up is None:
                    eta = torch.mean(torch.clamp(self.att_factor*(X_adv - X_nat), min=-self.epsilon, max=self.epsilon).detach_(), dim=0)
                    self.att_up = eta
                else:
                    eta = torch.mean(torch.clamp(self.att_factor*(X_adv - X_nat), min=-self.epsilon, max=self.epsilon).detach_(), dim=0)
                    self.att_up = self.att_up * self.momentum + eta * (1 - self.momentum)
                X = torch.clamp(X_nat + self.att_up, min=-1, max=1).detach_()

        attgan.G.zero_grad()
        return X, X - X_nat
    
    def universal_perturb_stargan(self, X_nat, y, c_trg, model):
        
        #复制原始数据并分离
        X = X_nat.clone().detach_()

        #进行迭代
        for i in range(self.k):
            X.requires_grad = True
            
            #应用多样性输入
            X_diverse=self.input_diversity(X)
            
            #生成输出
            output, feats = model(X_diverse, c_trg)
            model.zero_grad()
            
            #计算损失和梯度
            loss = self.loss_fn(output, y)
            loss.backward()
            grad = X.grad
            
            #更新对抗样本和扰动
            X_adv = X + self.a * grad.sign()
            if self.mode=='universal':
                if self.up is None:
                    eta = torch.mean(
                        torch.clamp(self.star_factor * (X_adv - X_nat), min=-self.epsilon, max=self.epsilon).detach_(),
                        dim=0)
                    self.up = eta
                else:
                    eta = torch.mean(torch.clamp(self.star_factor * (X_adv - X_nat), min=-self.epsilon, max=self.epsilon).detach_(),dim=0)
                    self.up = self.up * self.momentum + eta * (1 - self.momentum)
                X = torch.clamp(X_nat + self.up, min=-1, max=1).detach_()
            elif self.mode=='stargan':
                if self.star_up is None:
                    eta = torch.mean(
                        torch.clamp(self.star_factor * (X_adv - X_nat), min=-self.epsilon, max=self.epsilon).detach_(),
                        dim=0)
                    self.star_up = eta
                else:
                    eta = torch.mean(torch.clamp(self.star_factor * (X_adv - X_nat), min=-self.epsilon, max=self.epsilon).detach_(),dim=0)
                    self.star_up = self.star_up * self.momentum + eta * (1 - self.momentum)
                X = torch.clamp(X_nat + self.star_up, min=-1, max=1).detach_()
                
        model.zero_grad()
        return X, X - X_nat

    def universal_perturb_attentiongan(self, X_nat, y, c_trg, model):
        
        #复制原始数据并分离
        X = X_nat.clone().detach_()

        #进行迭代
        for i in range(self.k):
            X.requires_grad = True
            
            #应用多样性输入
            X_diverse=self.input_diversity(X)
            
            #生成输出
            output, _, _ = model(X_diverse, c_trg)
            model.zero_grad()
            
            #计算损失和梯度
            loss = self.loss_fn(output, y)
            loss.backward()
            grad = X.grad

            #更新对抗样本和扰动
            X_adv = X + self.a * grad.sign()
            if self.mode=='universal':
                if self.up is None:
                    eta = torch.mean(
                        torch.clamp(self.attention_factor * (X_adv - X_nat), min=-self.epsilon, max=self.epsilon).detach_(),
                        dim=0)
                    self.up = eta
                else:
                    eta = torch.mean(
                        torch.clamp(self.attention_factor * (X_adv - X_nat), min=-self.epsilon, max=self.epsilon).detach_(),
                        dim=0)
                    self.up = self.up * self.momentum + eta * (1 - self.momentum)
                X = torch.clamp(X_nat + self.up, min=-1, max=1).detach_()
            elif self.mode=='attention':
                if self.attention_up is None:
                    eta = torch.mean(
                        torch.clamp(self.attention_factor * (X_adv - X_nat), min=-self.epsilon, max=self.epsilon).detach_(),
                        dim=0)
                    self.attention_up = eta
                else:
                    eta = torch.mean(
                        torch.clamp(self.attention_factor * (X_adv - X_nat), min=-self.epsilon, max=self.epsilon).detach_(),
                        dim=0)
                    self.attention_up = self.attention_up * self.momentum + eta * (1 - self.momentum)
                X = torch.clamp(X_nat + self.attention_up, min=-1, max=1).detach_()                
        model.zero_grad()
        return X, X - X_nat

    def universal_perturb_HiSD(self, X_nat, transform, F, T, G, E, reference, y, gen, mask):#这个gen就是前面的方法中的model

        #复制原始数据并分离
        X = X_nat.clone().detach_()
         
        #进行迭代
        for i in range(self.k):
            X.requires_grad = True
            
             #应用多样性输入
            X_diverse=self.input_diversity(X)
            
            #生成输出
            c = E(X_diverse)
            c_trg = c
            s_trg = F(reference, 1)
            c_trg = T(c_trg, s_trg, 1)
            x_trg = G(c_trg)
            gen.zero_grad()

            #计算损失和梯度
            loss = self.loss_fn(x_trg, y)
            loss.backward()
            grad = X.grad

            #更新对抗样本和扰动
            X_adv = X + self.a * grad.sign()
            if self.mode=='universal':
                if self.up is None:
                    eta = torch.mean(
                        torch.clamp(self.HiSD_factor * (X_adv - X_nat), min=-self.epsilon, max=self.epsilon).detach_(),
                        dim=0)
                    self.up = eta
                else:
                    eta = torch.mean(
                        torch.clamp(self.HiSD_factor * (X_adv - X_nat), min=-self.epsilon, max=self.epsilon).detach_(),
                        dim=0)
                    self.up = self.up * self.momentum + eta * (1 - self.momentum)
                X = torch.clamp(X_nat + self.up, min=-1, max=1).detach_()
            elif self.mode=='HiSD':
                if self.HISD_up is None:
                    eta = torch.mean(
                        torch.clamp(self.HiSD_factor * (X_adv - X_nat), min=-self.epsilon, max=self.epsilon).detach_(),
                        dim=0)
                    self.HISD_up = eta
                else:
                    eta = torch.mean(
                        torch.clamp(self.HiSD_factor * (X_adv - X_nat), min=-self.epsilon, max=self.epsilon).detach_(),
                        dim=0)
                    self.HISD_up = self.HISD_up * self.momentum + eta * (1 - self.momentum)
                X = torch.clamp(X_nat + self.HISD_up, min=-1, max=1).detach_() 
        gen.zero_grad()
        return X, X - X_nat