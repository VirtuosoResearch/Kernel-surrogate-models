## Activate env
```
conda activate /pscratch/sd/j/jwl50/Hessian-view-of-grokking-code/.env
```

## Train
```
python main.py
# Old code:
python main.py --track_wandb --dataset reasoning --task needle_in_haystack

# New code:
python main_jerry.py --config ./src/configs/arithmetic_new.yaml

# Needle in haystack
python main_jerry.py --config ./src/configs/tasks/needle_in_haystack.yaml
python main_jerry.py --config ./src/configs/tasks/needle_in_haystack.yaml --opt_config ./src/configs/optimizers/nsm.yaml
python main_jerry.py --config ./src/configs/tasks/needle_in_haystack.yaml --opt_config ./src/configs/optimizers/sam.yaml

# Decimal addition
python main_jerry.py --config ./src/configs/tasks/decimal_addition.yaml
python main_jerry.py --config ./src/configs/tasks/decimal_addition.yaml --opt_config ./src/configs/optimizers/nsm.yaml
python main_jerry.py --config ./src/configs/tasks/decimal_addition.yaml --opt_config ./src/configs/optimizers/sam.yaml

# Parenthesis balancing
python main_jerry.py --config ./src/configs/tasks/parenthesis_balancing.yaml
python main_jerry.py --config ./src/configs/tasks/parenthesis_balancing.yaml --opt_config ./src/configs/optimizers/nsm.yaml
python main_jerry.py --config ./src/configs/tasks/parenthesis_balancing.yaml --opt_config ./src/configs/optimizers/sam.yaml
```