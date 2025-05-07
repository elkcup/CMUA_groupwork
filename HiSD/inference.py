# import packages
try:
    from core.utils import get_config
    from core.trainer import HiSD_Trainer
except:
    from HiSD.core.utils import get_config
    from HiSD.core.trainer import HiSD_Trainer
import argparse
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
import numpy as np
import time
from attacks import LinfPGDAttack

# use cpu by default

# device = 'cpu'

# load checkpoint

def prepare_HiSD(device=None):
    ## 增加兼容
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ## 增加兼容
    # device = 'cuda:0' 
    config = get_config('HiSD/configs/celeba-hq_256.yaml')
    noise_dim = config['noise_dim']
    image_size = config['new_size']
    checkpoint = 'HiSD/gen_00600000.pt'
    trainer = HiSD_Trainer(config)
    state_dict = torch.load(checkpoint, map_location=device)
    trainer.models.gen.load_state_dict(state_dict['gen_test'])
    trainer.models.gen.to(device)
  
    E = trainer.models.gen.encode
    T = trainer.models.gen.translate
    G = trainer.models.gen.decode
    M = trainer.models.gen.map
    F = trainer.models.gen.extract


    class ConditionalToTensor:
        def __call__(self, img):
            if isinstance(img, torch.Tensor):
                return img
            return transforms.ToTensor()(img)
    transform = transforms.Compose([transforms.Resize(image_size),
                                    ConditionalToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    reference = 'HiSD/examples/reference_glasses_2.jpg'
    reference = transform(Image.open(reference).convert('RGB')).unsqueeze(0).to(device)
    return transform, F, T, G, E, reference, trainer.models.gen


def inference_to_attack(x, transform, F, T, G, E, reference, gen):
    attack = LinfPGDAttack()
    with torch.no_grad():
        c = E(x)
        c_trg = c
        s_trg = F(reference, 1)
        c_trg = T(c_trg, s_trg, 1)
        x_trg = G(c_trg)
    attack.universal_perturb_HiSD(x.cuda(), transform, F, T, G, E, device, reference, x_trg, gen)






