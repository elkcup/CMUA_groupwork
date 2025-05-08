# CMUA-Others

目前进度：

1. 整体代码已上传完成
2. 正在整合代码

## PGD和MIM

1. 文件说明

   - 训练相关：

   | 文件 | 说明 |
   | ---- | ---- |
   |```mim.py```|MIM类及其对StarGAN、attentionGAN、HISD和AttGAN攻击方法的定义|
   |```PGD.py```|PGD类及其对StarGAN、attentionGAN、HISD和AttGAN攻击方法的定义|
   |```pgd_attack.py```|PGD在CeleBA数据集的前128张图片上进行训练|
   |```mim_attack.py```|MIM在CeleBA数据集的前128张图片上进行训练|
   |```best_params_pgd_mim.json```|目前搜索到PGD和MIM的极优参数|
   |```evaluate_pgd_mim.py```|定义评估PGD和MIM模型的方法|
   - 扰动相关：

     
   |文件|说明|
   |---|---|
   |```mim_stargan_perturbation.pt```| MIM针对StarGAN模型的扰动文件|
   |```mim_attgan_perturbation.pt```|MIM针对AttGAN模型的扰动文件|
   |```mim_hisd_perturbation.pt```| MIM针对HISD模型的扰动文件|
   |```mim_attentiongan_perturbation.pt```|MIM针对AttentionGAN模型的扰动文件|
   |```pgd_stargan_perturbation.pt```| PGD针对StarGAN模型的扰动文件|
   |```pgd_attgan_perturbation.pt```| PGD针对AttGAN模型的扰动文件|
   |```pgd_hisd_perturbation.pt```| PGD针对HISD模型的扰动文件|
   |```pgd_attentiongan_perturbation.pt```| PGD针对AttentionGAN模型的扰动文件|

2. 扰动训练</br>
   PGD：```python pgd_attack.py```</br>
   MIM：```python mim_attack.py```</br>

3. 推理：</br>

   


## DI2-FGSM and M-DI2-FGSM
1. 文件说明<br>
    ```fgsm_settings.json```：仅基于DI2-FGSM或仅基于M-DI2-FGSM训练扰动/推理时的参数配置文件<br>
    ```auto_fgsm_attacks.py```：用于定义DI2-FGSM/M-DI2-FGSM攻击器<br>
    ```auto_fgsm_evaluate.py```：基于DI2-FGSM/M-DI2-FGSM的扰动评估文件<br>
    ```auto_fgsm_inference.py```：基于DI2-FGSM/M-DI2-FGSM的扰动推理文件<br>
    ```auto_fgsm_model_data_prepare.py```：模型和数据准备文件<br>
    ```auto_fgsm_train.py```：用于基于DI2-FGSM或仅基于M-DI2-FGSM训练扰动的文件<br>
2. 扰动训练<br>
    如果需要基于DI2-FGSM训练通用扰动，请在```cmd```中运行```python auto_fgsm_train.py```，并在init_Attack函数中设置FGSMAttack的参数```mode='universal', method='DI2_FGSM'```；<br>
    如果需要基于DI2-FGSM训练某个模型的专用扰动，请在```cmd```中运行```python auto_fgsm_train.py```，并在init_Attack函数中设置FGSMAttack的参数```mode='模型名称', method='DI2_FGSM'```；<br>
    基于M-DI2-FGSM的情况同理。<br>
3. 推理<br>
    如果需要使用基于DI2-FGSM训练得到的通用扰动在测试数据上进行推理，请在```cmd```中运行```python auto_fgsm_inference.py```，并在init_Attack函数中设置FGSMAttack的参数```mode='universal', method='DI2_FGSM'```；<br>
    如果需要使用基于DI2-FGSM训练得到的某个模型的专用扰动在测试数据上进行推理，请在```cmd```中运行```python auto_fgsm_inference.py```，并在init_Attack函数中设置FGSMAttack的参数```mode='模型名称', method='DI2_FGSM'```；<br>
    基于M-DI2-FGSM的情况同理。<br>

## CMUA和AutoPGD

1. CMUA: CMUA的实现请参照CMUA的Readme.md文档，需要自行下载数据集和生成模型的权重
   + 1.1 属于CMUA的文件
   
     | 文件名      | 解释 |
        | ----------- | ----------- |
        | perturbation.py      | CMUA预先训练好的水印      |

3. APGD
    + 2.1 请将lyf文件夹下的这些文件放于项目根目录下

        | 文件名      | 解释 |
        | ----------- | ----------- |
        | auto_pgd_attacks.py      | 训练核心代码      |
        |auto_pgd_setting.json       |apgd设置文件        |
        | auto_pgd_perturbation.pt   | apgd预先训练好的水印        |
        |apgd_universal_attack.py       |    训练过程  |
        |apgd_universal_inference.py       |    推理过程  |
        |evaluate_cmua_apgd.py       |  评估cmua和apgd   |

    + 2.2 训练与推理
        + 训练：

            ```python
            python apgd_universal_attack.py
            ```

        + 推理：

            ```python
            python apgd_universal_inference.py
            ```

## BIM
1. 请将qxt文件夹下的这些文件放于项目根目录下   
```bim_attack.py```：bim训练核心代码<br>     
```bim_setting.json```：bim设置文件<br>           
```bim_perturbation.pt```：bim预先训练好的水印<br>           
```bim_universal_attack.py```：bim训练过程<br>     
```bim_universal_inference.py```：bim推理过程<br>     
```bim_evaluate.py```：评估bim 
2. 训练        
     ```python       
     python bim_universal_attack.py       
     ```
3. 推理：
     ```python       
     python bim_universal_inference.py       
     ```


## 对比试验结果查看

如果已经完成所有模型的训练，想要查看最终的对比结果请输入：

```python
python evaluate_all.py
```



## Citation

A cite from paper: CMUA-Watermark: A Cross-Model Universal Adversarial Watermark for Combating Deepfakes.

```
@inproceedings{huang2022cmua,
  title={Cmua-watermark: A cross-model universal adversarial watermark for combating deepfakes},
  author={Huang, Hao and Wang, Yongtao and Chen, Zhaoyu and Zhang, Yuze and Li, Yuheng and Tang, Zhi and Chu, Wei and Chen, Jingdong and Lin, Weisi and Ma, Kai-Kuang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={1},
  pages={989--997},
  year={2022}
}
```

## License

Just for academic research purposes
## Thanks

We use code from[CMAU](https://github.com/VDIGPKU/CMUA-Watermark), [StarGAN](https://github.com/yunjey/stargan), [GANimation](https://github.com/vipermu/ganimation), [pix2pixHD](https://github.com/NVIDIA/pix2pixHD), [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), [advertorch](https://github.com/BorealisAI/advertorch), [disrupting-deepfakes](https://github.com/natanielruiz/disrupting-deepfakes) and [nni](https://github.com/microsoft/nni). These are all great repositories and we encourage you to check them out and cite them in your work.
