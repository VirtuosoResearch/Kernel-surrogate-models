import torch
import torch.nn.functional as F
import random

from src.data.task import Task
from src.configs.config import ConfigDict
from src.data.util import split_data

class NeedleInHaystack(Task):

    def __init__(self, config):
        super().__init__(config)
        self.max_len = config.max_len
        self.vocab_size = config.vocab_size
        self.dataset_size = config.dataset_size
        self.split_ratio = config.split_ratio

        self.get_data()

    def get_model_hparams(self):
        return ConfigDict({
            "num_tokens": self.vocab_size,
            "max_len": self.max_len * 2 + 3,
        })

    # Output: train_data, valid_data
    def get_data(self):
        data = needle_in_haystack(
            max_len=self.max_len,
            vocab_size=self.vocab_size,
            dataset_size=self.dataset_size,
        )
        self.train_data, self.valid_data = split_data(data, self.split_ratio)
        self.train_num = self.train_data.shape[0]
        self.valid_num = self.valid_data.shape[0]
        return self.train_data, self.valid_data

    def get_loss(self):
        # logits: (B, L-1, C), sequence: (B, L)
        def loss_fn(logits, sequence):
            wanted = sequence[:, :-1] >= self.max_len + self.vocab_size//2 # (B, L-1)
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
            wanted = sequence[:, :-1] >= self.max_len + self.vocab_size//2 # (B, L-1)
            corr = (predictions == sequence[:, 1:]) * wanted # (B, L-1)
            return {
                "accuracy": corr.sum().item() / wanted.sum().item()
            }
        return compute_metrics

def needle_in_haystack(max_len, vocab_size, dataset_size):
    def gen_single():
        seq_len = random.randint(1,max_len)
        keys = list(range(max_len))
        random.shuffle(keys)
        keys = keys[:seq_len]
        values = [random.randint(1,vocab_size//2-1) for _ in range(seq_len)]
        query = random.choice(keys)
        l = sum(([a+vocab_size//2,b] for a,b in zip(keys,values)),[])+[query+max_len+vocab_size//2]
        l = l + [values[keys.index(query)]]
        l += [0] * (max_len*2+3-len(l))
        assert max(l)<vocab_size and min(l)>=0
        return l

    # Generate dataset
    tasks = []
    for i in range(dataset_size):
        tasks.append(gen_single())
    tasks = torch.tensor(tasks) # (size, max_len*2+3)

    return tasks