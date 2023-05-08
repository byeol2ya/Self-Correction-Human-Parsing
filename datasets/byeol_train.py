
import os
import wandb

import networks

def main():
    wandb.init(project="byl-inpainting")
    print('CUDA_HOME : ', os.environ.get('CUDA_HOME'))
    
    
    AugmentCE2P = networks.init_model('resnet101', num_classes=20)
    
    
    return 0