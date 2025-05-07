"""
PGD类实现
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# region ver1
class PGDAttack(object):
    def __init__(self, model=None, device=None, epsilon=0.05, iterations=10, step_size=0.01):
        """
        标准PGD攻击实现
        
        参数:
            model: 目标模型
            device: 计算设备
            epsilon: 扰动大小上限
            iterations: 迭代次数
            step_size: 每步梯度步长
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.epsilon = epsilon
        self.iterations = iterations
        self.step_size = step_size
        self.loss_fn = nn.MSELoss().to(device)
        self.device = device

    def set_params(self, epsilon=None, step_size=None, iterations=None):
        """
        设置攻击参数
        
        参数:
            epsilon: 扰动大小上限
            step_size: 每步梯度步长
        """
        if epsilon is not None:
            self.epsilon = epsilon
        if step_size is not None:
            self.step_size = step_size
        if iterations is not None:
            self.iterations = iterations

    def get_params(self):
        """
        获取当前攻击参数
        
        返回:
            epsilon: 扰动大小上限
            step_size: 每步梯度步长
        """
        return {'epsilon': self.epsilon, 'step_size': self.step_size,'iterations': self.iterations}
    
    def perturb(self, X_nat, y, model=None, c_trg=None):
        """
        执行PGD攻击
        
        参数:
            X_nat: 原始输入样本
            y: 目标输出
            c_trg: 目标域标签（可选）
            
        返回:
            X: 对抗样本
            delta: 添加的扰动
        """
        if model is not None:
            self.model = model


        # 随机初始化
        X_nat=X_nat.clone().detach_().to(self.device)
        X = X_nat.clone().detach_().to(self.device)
        eps = torch.clamp(torch.tensor(
            np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')
        ), min=-self.epsilon, max=self.epsilon).to(self.device)
        X = X_nat + eps

        for i in range(self.iterations):
            X.requires_grad = True
            
            # 前向传播
            if c_trg is not None:
                output = self.model(X, c_trg)
                output = output[0] if isinstance(output, tuple) or isinstance(output,list) else output
            else:
                output = self.model(X)
                output = output[0] if isinstance(output, tuple) or isinstance(output, list) else output

            # print(f"模型输出类型: {type(output)}")
 
            
            # 计算损失
            self.model.zero_grad()
            loss = self.loss_fn(output, y)
            loss.backward()
            
            # 梯度更新
            grad = X.grad
            X_adv = X + self.step_size * grad.sign()
            X_adv = torch.clamp(X_adv, min=X_nat - self.epsilon, max=X_nat + self.epsilon)
            X = X_adv.detach_()
            
        return X, X - X_nat
    
    def perturb_stargan(self, x_ori, y, c_trg, model):
        """
        对StarGAN模型的攻击
        
        参数:
            x_ori: 原始输入图像
            y: 目标输出图像
            c_trg: 目标域标签
            model: StarGAN模型实例
            
        返回:
            x_adv: 生成的对抗样本
            perturb: 添加到原始图像的扰动
        """
        model_backup = self.model
        self.model = model

        x_adv, perturb = self.perturb(x_ori, y, model, c_trg)
        self.model = model_backup
        return x_adv, perturb
    
    def perturb_attentiongan(self, x_ori, y, c_trg, model):
        """
        对AttentionGAN模型的攻击
        
        参数:
            x_ori: 原始输入图像
            y: 目标输出图像
            c_trg: 目标域标签
            model: AttentionGAN模型实例
            
        返回:
            x_adv: 生成的对抗样本
            perturb: 添加到原始图像的扰动
        """

        model_backup = self.model
        self.model = model
        x_adv, perturb = self.perturb(x_ori, y, model, c_trg)
        self.model = model_backup
        return x_adv, perturb
    
    def perturb_attgan(self, x_ori, x_att, y, attgan):
        """
        对AttGAN模型的攻击
        
        参数:
            x_ori: 原始输入图像
            x_att: 属性向量，指定要改变的人脸属性
            y: 目标输出图像
            attgan: AttGAN模型实例
            
        返回:
            x_adv: 生成的对抗样本
            perturb: 添加到原始图像的扰动
        """

        X = x_ori.clone().detach_().to(self.device)
        eps = torch.clamp(torch.tensor(
            np.random.uniform(-self.epsilon, self.epsilon, x_ori.shape).astype('float32')
        ), min=-self.epsilon, max=self.epsilon).to(self.device)
        X = x_ori + eps

        for i in range(self.iterations):
            X.requires_grad = True
            
            # 前向传播
            if x_att is not None:
                output = attgan.G(X, x_att)
                output = output[0] if isinstance(output, tuple) or isinstance(output,list) else output
            else:
                output = attgan.G(X)
                output = output[0] if isinstance(output, tuple) or isinstance(output, list) else output

            # print(f"模型输出类型: {type(output)}")
 
            
            # 计算损失
            attgan.G.zero_grad()
            loss = self.loss_fn(output, y)
            loss.backward()
            
            # 梯度更新
            grad = X.grad
            X_adv = X + self.step_size * grad.sign()
            X_adv = torch.clamp(X_adv, min=x_ori - self.epsilon, max=x_ori + self.epsilon)
            X = X_adv.detach_()
           
        return X, X - x_ori
        
    def perturb_hisd(self, x_ori, reference, y, E, F, T, G, gen,mask=None, transform=None):
        """
        对HiSD模型的攻击
        
        参数:
            x_ori: 原始输入图像
            reference: 参考图像，用于提取风格
            y: 目标输出图像
            E: HiSD的内容编码器
            F: HiSD的风格编码器
            T: HiSD的风格转换器
            G: HiSD的生成器
            gen: 生成器模型
            mask: 可选，掩码矩阵
            transform: 可选，在攻击前后应用的转换函数
            
        返回:
            x_adv: 对抗样本
            perturb: 扰动
        """
        batch_size = x_ori.size(0)
        X = x_ori.clone().detach_().to(self.device)
        eps = torch.clamp(torch.tensor(
            np.random.uniform(-self.epsilon, self.epsilon, x_ori.shape).astype('float32')
        ), min=-self.epsilon, max=self.epsilon).to(self.device)
        X = x_ori + eps
   
        for i in range(self.iterations):
            if transform is not None:
                X = transform(X)
            X.requires_grad = True

            if reference.size(0) == 1 and batch_size > 1:
                reference_batch = reference.expand(batch_size, -1, -1, -1)
            else:
                reference_batch = reference

            c = E(X)
            c_trg = c
            s_trg = F(reference_batch, 1)
            c_trg = T(c_trg, s_trg, 1)
            x_trg = G(c_trg)
            gen.zero_grad()

            loss = self.loss_fn(x_trg, y)
            loss.backward()
            
            # 梯度更新
            grad = X.grad
            X_adv = X + self.step_size * grad.sign()
            X_adv = torch.clamp(X_adv, min=x_ori - self.epsilon, max=x_ori + self.epsilon)
            X = X_adv.detach_()

        perturb = X - x_ori


        if mask is not None:
            # 处理掩码的批量操作
            if mask.dim() == 2: 
                mask = mask.unsqueeze(0).unsqueeze(0) 
                mask = mask.expand(batch_size, 3, -1, -1)  
            elif mask.dim() == 3: 
                mask = mask.unsqueeze(1) 
                mask = mask.expand(-1, 3, -1, -1) 
                
            masked_perturb = perturb * mask
            masked_x_adv = x_ori + masked_perturb
            return masked_x_adv, masked_perturb
        
        return X, X - x_ori
# endregion

