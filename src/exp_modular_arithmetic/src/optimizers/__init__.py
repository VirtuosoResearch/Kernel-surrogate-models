import torch

# Optimizers: NSM, SAM
from src.optimizers.nsm import NSM
from src.optimizers.sam import SAM

optimizer_names = {
    "NSM": NSM,
    "SAM": SAM,
}

def get_optimizer(optimizer_config, model):
    
    base_optimizer = getattr(torch.optim, optimizer_config.base_optimizer) # AdamW by default
    if optimizer_config.name in optimizer_names.keys():
        print(f"Using optimizer {optimizer_config.name}")
        optimizer_type = optimizer_names[optimizer_config.name]
        optimizer = optimizer_type(
            model.parameters(),
            base_optimizer,
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.wd,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            **optimizer_config.optimizer_kwargs
        )
    else:
        print(f"Using default optimizer {optimizer_config.base_optimizer}")
        optimizer = base_optimizer(
            model.parameters(),
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.wd
        )

    return optimizer