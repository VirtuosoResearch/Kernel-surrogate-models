import torch
import torch.nn.functional as F
import itertools
import numpy as np
from sympy.combinatorics.permutations import Permutation

from src.data.task import Task
from src.configs.config import ConfigDict
from src.data.util import split_data

class ModularArithmetic(Task):

    def __init__(self, config):
        super().__init__(config)
        self.p = config.p
        self.task = config.task
        self.num_input_numbers = config.num_input_numbers
        self.num_total_samples = config.num_total_samples
        self.train_ratio = config.train_ratio
        self.valid_ratio = config.valid_ratio
        self.seq_len = config.num_input_numbers * 2 + 1 # config.seq_len # deprecated 
        self.split_ratio = config.split_ratio # deprecated

        self.get_data()

        self.num_samples_for_neighbors = config.num_samples_for_neighbors
        if self.num_samples_for_neighbors > 0:
            self.get_neighbor_data()
        else:
            self.neighbor_data = None

    def get_model_hparams(self):
        return ConfigDict({
            "num_tokens": self.p + 2,
            "max_len": 5,
        })

    # Output: train_data, valid_data
    def get_data(self):
        # data = mod_p_data(self.p, self.task)
        data = generate_data(self.num_input_numbers, self.num_total_samples, task=self.task, value_range=(0, self.p))
        print("The total number of data points is: ", data.shape[0])

        # self.train_data, self.valid_data = split_data(data, self.split_ratio)
        self.train_data = data[:int(data.shape[0] * self.train_ratio)]
        self.valid_data = data[-int(data.shape[0] * self.valid_ratio):]
        self.train_num = self.train_data.shape[0]
        self.valid_num = self.valid_data.shape[0]
        print(f"Train data: {self.train_data.shape}", f"Valid data: {self.valid_data.shape}")
        return self.train_data, self.valid_data

    def get_neighbor_data(self, perturb_pos=3):
        self.neighbor_data = []
        for i in range(perturb_pos):
            neighbor_data = generate_neighbor_data(self.train_data, num_samples=self.num_samples_for_neighbors,
                                                    num_input_numbers=self.num_input_numbers, task=self.task, value_range=(0, self.p), hop=i+1)
            self.neighbor_data.append(neighbor_data)
        self.neighbor_num = [neighbor_data.shape[0] for neighbor_data in self.neighbor_data]

    def get_neighbor_dataloader(self, batch_size):
        assert self.neighbor_data is not None
        neighbor_dataloaders = []
        for neighbor_data in self.neighbor_data:
            neighbor_dataset = torch.utils.data.TensorDataset(neighbor_data)
            neighbor_dataloader = torch.utils.data.DataLoader(
                neighbor_dataset,
                batch_size=batch_size,
                shuffle=False
            )
            neighbor_dataloaders.append(neighbor_dataloader)
        return neighbor_dataloaders

    def get_loss(self):
        # logits: (B, L-1, C), sequence: (B, L)
        def loss_fn(logits, sequence):
            return F.cross_entropy(logits[:, -1], sequence[:, -1])
        return loss_fn

    def get_eval_metrics(self):
        # logits: (B, L-1, C), sequence: (B, L)
        def compute_metrics(logits, sequence):
            predictions = logits[:, -1].argmax(dim=-1)
            return {
                "accuracy": (predictions == sequence[:, -1]).float().mean().item()
            }
        return compute_metrics
    
    def distribution_shift(self):
        def KL_divergence(a, b):
            return torch.sum(a * torch.log(a / b))
        mean_distribution = torch.ones(self.p) / self.p
        train_count = torch.bincount(self.train_data[:, -1])
        train_distribution = train_count / train_count.sum()
        valid_count = torch.bincount(self.valid_data[:, -1])
        valid_distribution = valid_count / valid_count.sum()
        KL_train_valid = KL_divergence(train_distribution, valid_distribution)
        KL_train_mean = KL_divergence(train_distribution, mean_distribution)
        KL_valid_mean = KL_divergence(valid_distribution, mean_distribution)
        print(f"Distribution shift of task: {self.task} {KL_train_valid}")
        print(KL_train_mean)
        print(KL_valid_mean)

# Compute y^exp (mod p)
# y: tensor
# exp, p: int
def mod_exp(y, exp, p):
    result = torch.ones_like(y)
    base = y % p
    rem = exp
    while rem > 0:
        if rem % 2 == 1:
            result = (result * base) % p
        base = (base * base) % p
        rem = rem // 2
    return result

def generate_task_results(sequences, num_input_numbers, p, task):
    eq_token = p
    op_token = p + 1

    if task == "quad1":
        result = torch.zeros_like(sequences[:, 0])
        for i in range(num_input_numbers):
            result += sequences[:, i] ** 2 
            if i < num_input_numbers - 1:
                result += sequences[:, i] * sequences[:, (i+1)]
        result = result % p
    elif task == "addition":
        result = torch.zeros_like(sequences[:, 0])
        for i in range(num_input_numbers):
            result += sequences[:, i]
        result = result % p
    elif task == "mul":
        result = torch.ones_like(sequences[:, 0])
        for i in range(num_input_numbers):
            result = (result * sequences[:, i]) 
        result = result % p
    elif task == "sub":
        result = sequences[:, 0]
        for i in range(1, num_input_numbers):
            result = (result - sequences[:, i]) 
        result = result % p
    else:
        raise NotImplementedError

    eq = torch.ones_like(sequences[:, 0]) * eq_token
    op = torch.ones_like(sequences[:, 0]) * op_token

    # insert eq between every number in a sequence
    data = [[sequences[:, i], eq] for i in range(num_input_numbers-1)]
    data = list(itertools.chain(*data)) + [sequences[:, -1]] + [op] + [result]
    return torch.stack(data).T

def generate_neighbor_data(train_data, num_samples, num_input_numbers=2, value_range=(0, 97), task="quad1", hop=1):
    torch.manual_seed(42)
    sampled_data = train_data[torch.randperm(train_data.shape[0])][:num_samples]
    # generate neighbors
    #   for x o y = z
    #   we change either x and y, then regenerate z
    sampled_data = torch.stack([sampled_data[:, 2*i] for i in range(num_input_numbers)])
    sequences = []

    assert num_input_numbers == 2 # now only support 2 numbers
    for i in range(hop+1):
        j = hop - i
        # i represent how much to change number x, j represent how much to change number y
        delta_xs = [i, -i]  if i != 0 else [0]
        delta_ys = [j, -j]  if j != 0 else [0]
        for (delta_x, delta_y) in itertools.product(delta_xs, delta_ys):
            new_data = sampled_data.clone()
            if delta_x != 0:   
                new_data[0] = (new_data[0] + torch.ones_like(new_data[0]) * delta_x) % value_range[1]
            if delta_y != 0:
                new_data[1] = (new_data[1] - torch.ones_like(new_data[1]) * delta_y) % value_range[1]
            sequences.append(new_data.T)
    
    sequences = torch.concat(sequences, dim=0)
    data = generate_task_results(sequences, num_input_numbers, p=value_range[1], task=task)
    return data

def generate_data(num_input_numbers=2, num_total_samples=9600, value_range=(0, 97), task="quad1"):
    """
    Generate a tensor of unique sequences where each number in the sequence is between 0 and 97.
    
    :param num_total_samples: Total number of sequences to generate.
    :param sequence_length: Fixed length of each sequence.
    :param value_range: Range of numbers (default is 0 to 97).
    :return: Torch tensor of unique generated sequences.
    """

    # enable the generation of multiple numbers 
    if num_input_numbers == 2: 
        x = torch.arange(value_range[0], value_range[1])
        y = torch.arange(value_range[0], value_range[1])
        sequences = torch.cartesian_prod(x, y) # shape: (2, 9409)
    elif num_input_numbers == 3:
        x = torch.arange(value_range[0], value_range[1])
        y = torch.arange(value_range[0], value_range[1])
        z = torch.arange(value_range[0], value_range[1])
        sequences = torch.cartesian_prod(x, y, z) # shape: (3, 912673)
    elif num_input_numbers >= 4:
        # do not de-duplicate since the probability is low
        torch.manual_seed(42)
        sequences = torch.randint(value_range[0], value_range[1], (num_total_samples, num_input_numbers))

    # set manual seed for reproducibility
    torch.manual_seed(33)
    if num_total_samples < len(sequences):
        sequences = sequences[torch.randperm(sequences.shape[0])][:num_total_samples]
    else:
        sequences = sequences[torch.randperm(sequences.shape[0])]
    sequences = sequences.type(torch.int64) # avoid overflow

    data = generate_task_results(sequences, num_input_numbers, p=value_range[1], task=task)

    return data

def mod_p_data(p, task="multiplication",
    num_input_numbers=2, # number of numbers in the input polynomial equations
    num_total_samples=10000): # number of total samples to generate
    """x◦y = x/y (mod p) for 0 ≤ x < p, 0 < y < p
    """

    # tokens for <op> and <=>. It's not clear why <=> is needed at all since it
    # has no effect on the output, but we'll leave it in to best follow the
    # paper.
    eq_token = p
    op_token = p + 1

    x = torch.arange(p)
    y = torch.arange(p)
    x, y = torch.cartesian_prod(x, y).T

    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token
    
    if task == "mul":
        result = (x * y) % p
    elif task == "addition":
        result = (x + y) % p
    elif task == "sub":
        result = (x - y) % p
    elif task == "div": 
        x = torch.arange(p)
        y = torch.arange(1, p)
        x, y = torch.cartesian_prod(x, y).T
        eq = torch.ones_like(x) * eq_token
        op = torch.ones_like(x) * op_token
        result = x
        x = y*result % p
    elif task == "parity_division": # TODO JL fix
        x = torch.arange(p)
        y_odd = torch.arange(1, p, step=2)
        y_even = torch.arange(0, p, step=2)
        x_1, y_odd = torch.cartesian_prod(x, y_odd).T
        x_2, y_even = torch.cartesian_prod(x, y_even).T
        result_1 = x_1
        x_1 = y_odd*result_1 % p
        result_2 = (x_2 - y_even) % p
        x = torch.cat((x_1, x_2))
        y = torch.cat((y_odd, y_even))
        result = torch.cat((result_1, result_2))
    elif task == 'xonly':
        result = x
    elif task == 'x2only':
        result = x**2 % p
    elif task == 'x3only':
        result = x**3 % p
    elif task == 'x4only':
        result = x**4 % p
    elif task == "sum_of_squares":
        result = (x**2 + y**2) % p
    elif task == "quad1":
        result = (x**2 + x*y + y**2) % p
    elif task == "quad2":
        result = (x**2 + x*y + y**2 + x) % p
    elif task == "cubic1":
        result = (x**3 + x*y) % p
    elif task == "cubic2":
        result = (x**3 + x*(y**2) + y) % p
    elif task == "cubic3":
        result = (x**3 + x*(y**2) + y + y**3) % p
    elif task == 'add2':
        result = (x**2 + y**2) % p
    elif task == 'add3':
        result = (x**3 + y**3) % p
    elif task == 'add5':
        result = (x**5 + y**5) % p
    elif task == 'add4':
        result = (x**4 + y**4) % p
    elif task == 'add6':
        result = (x**6 + y**6) % p
    elif task == 'quada': 
        result = (x**2 + x*y + y**2 + x) % p
    elif task == 'quadb': 
        result = (x**2 + x*y + y**2 + y) % p
    elif task == 'quadab':
        result = (x**2 + x*y + y**2 + x + y) % p

    # "All of our experiments used a small transformer trained on datasets of
    # equations of the form a◦b = c, where each of “a”, “◦”, “b”, “=”, and “c”
    # is a seperate token"
    return torch.stack([x, op, y, eq, result]).transpose(0, 1) # (N, L)
    #return torch.stack([x, op, y, eq, result])
