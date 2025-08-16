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

def train(model, loader, lr=0.4, epochs=24, momentum=0.9,
          weight_decay=5e-4, lr_peak_epoch=5, label_smoothing=0.0, model_id=0):

    opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    iters_per_epoch = len(loader)
    # Cyclic LR with single triangle
    lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
                            [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                            [0, 1, 0])
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)

    loop = tqdm(range(epochs), desc='Training epochs')
    for ep in loop:
        for it, (ims, labs) in enumerate(loader):
            ims = ims.cuda()
            labs = labs.cuda()
            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda"):
                out = model(ims)
                loss = loss_fn(out, labs)

            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            loop.set_description(f'Epoch {ep+1}/{epochs} | Loss: {loss.item():.4f} | LR: {opt.param_groups[0]["lr"]:.6f}')
        #if ep in [23]:
        #    torch.save(model.state_dict(), f'./checkpoints/sd_{model_id}_epoch_{ep}.pt')

    return model

def evaluate(model, loader):
    model.eval()
    loss_fn = CrossEntropyLoss()
    correct = 0
    total = 0
    loss_list = []
    with torch.no_grad():
        for ims, labs in loader:
            ims = ims.cuda()
            labs = labs.cuda()
            out = model(ims)
            preds = out.argmax(dim=1)
            loss = loss_fn(out, labs)
            loss_list.append(loss.item())
            correct += (preds == labs).sum().item()
            total += labs.size(0)
    acc = correct / total
    avg_loss = sum(loss_list) / len(loss_list) if loss_list else 0
    return float(acc), float(avg_loss)

os.makedirs('./checkpoints', exist_ok=True)
train_loaders, test_loaders = get_subset_dataloader(batch_size=512, shuffle=True)

for i in range(len(train_loaders)):
    print(f'Training model {i} on subset {i} ...')
    model = construct_rn9().to(memory_format=torch.channels_last).cuda()
    model = train(model, train_loaders[i], model_id=i, epochs=25)
    # save model and results to json
    torch.save(model.state_dict(), f'./checkpoints/model_{i}.pt')
    results = {
        'model_id': i,
        'test_loss': [],  # will be filled after evaluation
        'test_accuracy': [],  # will be filled after evaluation
    }
    print(f'Evaluating model {i} on all test splits ...')
    for j in range(len(test_loaders)):
        acc, avg_loss = evaluate(model, test_loaders[j])
        print(f'  Test on split {j}: Acc {acc*100:.2f}%, Avg Loss {avg_loss:.4f}')
        results['test_loss'].append(avg_loss)
        results['test_accuracy'].append(acc)
    with open(f'./results/real/results_{i}.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(f'Model {i} results saved to ./results/real/results_{i}.json')

