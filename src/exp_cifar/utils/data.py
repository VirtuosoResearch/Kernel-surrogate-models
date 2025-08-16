import torch
from torch.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
from torch.optim import SGD, lr_scheduler
import torchvision

def get_dataloader(batch_size=256, num_workers=8, split='train', shuffle=False, augment=True):
    if augment:
        transforms = torchvision.transforms.Compose(
                        [torchvision.transforms.RandomHorizontalFlip(),
                         torchvision.transforms.RandomAffine(0),
                         torchvision.transforms.ToTensor(),
                         torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                          (0.2023, 0.1994, 0.201))])
    else:
        transforms = torchvision.transforms.Compose([
                         torchvision.transforms.ToTensor(),
                         torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                          (0.2023, 0.1994, 0.201))])

    is_train = (split == 'train')
    dataset = torchvision.datasets.CIFAR10(root='/data/shared/cifar/',
                                           download=True,
                                           train=is_train,
                                           transform=transforms)

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         shuffle=shuffle,
                                         batch_size=batch_size,
                                         num_workers=num_workers)

    return loader