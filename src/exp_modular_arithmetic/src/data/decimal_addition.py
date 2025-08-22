import torch
import torch.nn.functional as F
import random

from src.data.task import Task
from src.configs.config import ConfigDict
from src.data.util import split_data

class DecimalAddition(Task):

    def __init__(self, config):
        super().__init__(config)
        self.min_len = config.min_len
        self.max_len = config.max_len
        self.vocab_size = 31
        self.dataset_size = config.dataset_size
        self.split_ratio = config.split_ratio

        self.get_data()

    def get_model_hparams(self):
        return ConfigDict({
            "num_tokens": self.vocab_size,
            "max_len": self.max_len * 3 + 4,
        })

    # Output: train_data, valid_data
    def get_data(self):
        data = decimal_addition(
            min_len=self.min_len,
            max_len=self.max_len,
            dataset_size=self.dataset_size,
        )
        self.train_data, self.valid_data = split_data(data, self.split_ratio)
        self.train_num = self.train_data.shape[0]
        self.valid_num = self.valid_data.shape[0]
        return self.train_data, self.valid_data

    def get_loss(self):
        # logits: (B, L-1, C), sequence: (B, L)
        def loss_fn(logits, sequence):
            wanted = sequence[:, :-1] >= 20 # (B, L-1)
            labels = sequence[:, 1:] # (B, L-1)
            # Mask
            logits_masked = logits.reshape(-1, logits.shape[-1])[wanted.view(-1)]
            labels_masked = labels.reshape(-1)[wanted.view(-1)]
            # Cross entropy
            loss = F.cross_entropy(
                logits_masked,
                labels_masked,
            )
            return loss
        return loss_fn

    def get_eval_metrics(self):
        # logits: (B, L-1, C), sequence: (B, L)
        def compute_metrics(logits, sequence):
            predictions = logits.argmax(dim=-1) # (B, L-1)
            wanted = sequence[:, :-1] >= 20 # (B, L-1)
            corr = (predictions == sequence[:, 1:]) * wanted
            return {
                "accuracy": corr.sum().item() / wanted.sum().item()
            }
        return compute_metrics

def decimal_addition(min_len, max_len, dataset_size):
    def drawnum():
        o = random.randint(min_len,max_len)
        return random.randint(10**(o-1),10**o-1)

    max_lenl = max_len * 3 + 4
    def gen_single():
        a = drawnum()
        b = drawnum()
        L = str(a)[::-1]+'+'+str(b)[::-1]+'='
        R = str(a+b)[::-1]
        l = [int(x) if '0'<=x<='9' else {'+':10,'=':11}[x] for x in L] + [int(x)+20 for x in R] + [30] # 30=stop
        assert len(l) <= max_lenl
        l+=[12]*(max_lenl-len(l))
        return l

    # Generate dataset
    tasks = []
    for i in range(dataset_size):
        tasks.append(gen_single())
    tasks = torch.tensor(tasks) # (dataset_size, max_len * 3 + 4)

    return tasks