import torch

# Optimizers: NSM, SAM
from src.optimizers.nsm import NSM
from src.optimizers.sam import SAM

# TODO JL 10/25/24 restore lr scheduler
def get_optimizer(args, model):

    # For most experiments we used AdamW optimizer with learning rate 10−3,
    # weight decay 1, β1 = 0.9, β2 = 0.98
    if args.nsm:
        # NSM optimizer
        print(f"Using NSM optimizer with sigma={args.nsm_sigma}, distribution={args.nsm_distribution}")
        base_optimizer = getattr(torch.optim, args.optimizer)
        optimizer = NSM(
            model.parameters(),
            base_optimizer,
            sigma=args.nsm_sigma,
            distribution=args.nsm_distribution,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
        )
        #  linear learning rate warmup over the first 10 updates
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer.base_optimizer, lambda update: 1 if update > 10 else update / 10
        # )
    elif args.sam:
        # SAM optimizer
        print(f"Using SAM optimizer")
        base_optimizer = getattr(torch.optim, args.optimizer)
        optimizer = SAM(
            model.parameters(),
            base_optimizer,
            rho=args.sam_rho,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
        )
        #  linear learning rate warmup over the first 10 updates
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer.base_optimizer, lambda update: 1 if update > 10 else update / 10
        # )
    else:
        # AdamW optimizer
        optimizer = getattr(torch.optim, args.optimizer)(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
        )
        #  linear learning rate warmup over the first 10 updates
        #scheduler = torch.optim.lr_scheduler.LambdaLR(
        #    optimizer, lambda update: 1 if update > 10 else update / 10
        #)

    return optimizer