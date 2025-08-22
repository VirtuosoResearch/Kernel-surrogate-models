import torch
from tqdm import tqdm
from torch import Tensor, LongTensor
from typing import Tuple, List, Dict, Any, Union, Optional

def make_unary_operation_data(operator: str, operands: Tensor) -> List[str]:
    """
    :param operator: The unary operator to apply to each operand e.g. '+'
    :param operands: A tensor of operands
    :returns: list of equations"""
    num_examples = len(operands)

    if operator == "sort":
        rhs = torch.sort(operands, dim=1)[0]
    elif operator == "reverse":
        rhs = torch.flip(operands, dims=(1,))
    elif operator == "copy":
        rhs = operands
    else:
        raise Exception("unsupported operator")

    def func(L, R):
        L = map(str, L)
        R = map(str, R)
        return f"{operator} {' '.join(L)} = {' '.join(R)}"

    if num_examples < 1000000000:
        eqs = [
            func(L, R)
            for L, R in tqdm(
                zip(operands.tolist(), rhs.tolist()), total=num_examples
            )
        ]
    else:
        with ProcessPoolExecutor() as executor:
            eqs = executor.map(func, tqdm(zip(operands, rhs), total=num_examples))

    return eqs