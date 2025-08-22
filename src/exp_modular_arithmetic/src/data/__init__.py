from src.data.modp import ModularArithmetic
from src.data.needle_in_haystack import NeedleInHaystack
from src.data.decimal_addition import DecimalAddition
from src.data.parenthesis_balancing import ParenthesisBalancing

task_names = {
    "modular_arithmetic": ModularArithmetic,
    "needle_in_haystack": NeedleInHaystack,
    "decimal_addition": DecimalAddition,
    "parenthesis_balancing": ParenthesisBalancing,
}

def get_task(task_config):
    if task_config.name not in task_names:
        raise ValueError(f"Unknown task {task_config.name}")
    return task_names[task_config.name](task_config.task_kwargs)