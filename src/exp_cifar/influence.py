import json
import os
from pathlib import Path
import wget
from tqdm import tqdm
import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
from torch.optim import SGD, lr_scheduler
import torchvision
from tqdm import tqdm

from utils.data import get_dataloader, get_subset_dataloader
from utils.models import construct_rn9
import pytorch_influence_functions as ptif

if __name__ == "__main__":
    config = ptif.get_default_config()
    model = construct_rn9().to(memory_format=torch.channels_last).cuda()
    model.load_state_dict(torch.load('./checkpoints/model_0.pt'))
    train_loader = get_dataloader(batch_size=256, num_workers=8, split='train', shuffle=False, augment=False)
    test_loader = get_dataloader(batch_size=256, num_workers=8, split='test', shuffle=False, augment=False)
    ptif.init_logging('logfile.log')
    ptif.calc_img_wise(config, model, train_loader, test_loader)