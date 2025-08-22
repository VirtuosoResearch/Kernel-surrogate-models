import torch
import torch.nn.functional as F
import random

from src.data.task import Task
from src.configs.config import ConfigDict
from src.data.util import split_data

class ParenthesisBalancing(Task):
    
    def __init__(self, config):
        super().__init__(config)
        self.max_len = config.max_len
        self.vocab_size = 4
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
        data = parenthesis_balancing(
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
            wanted = sequence[:, :-1] == self.vocab_size - 1 # (B, L-1)
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
            wanted = sequence[:, :-1] == self.vocab_size - 1 # (B, L-1)
            corr = (predictions == sequence[:, 1:]) * wanted # (B, L-1)
            return {
                "accuracy": corr.sum().item() / wanted.sum().item()
            }
        return compute_metrics

def parenthesis_balancing(max_len, dataset_size):
    vocab_size = 4

    def gen_balanced(t):
        assert t>=0
        if t==0: return ''
        if t==1: return '()'
        if random.randint(0,1): return '('+gen_balanced(t-1)+')'
        u = random.randint(1,t-1)
        return gen_balanced(u)+gen_balanced(t-u)

    def is_balanced(l):
        s = 0
        for c in l:
            if c=='(':
                s += 1
            elif c==')':
                if s<=0: return False
                s -= 1
        return s==0

    def gen_single():
        # 0: EOF, 1: '(', 2: ')', 3: '?'
        assert 3 == vocab_size-1 # position of '?'

        mutate = random.randint(0,3)
        if random.randint(0,2)==0:
            seq_len = random.randint(1,max_len*2)
            l = ''.join([random.choice('()') for _ in range(seq_len)])
        else:
            # random balanced
            l = gen_balanced(random.randint(1,max_len))
            if random.randint(0,1):
                mutate = 0  # 1/3 chance of correct + no mutation
        if mutate&1:
            rep = 1
            while random.randint(0,1): rep += 1
            for _ in range(rep):
                p = random.randint(0,len(l)-1)
                q = random.randint(0,len(l)-1)
                if p>=q: continue
                # swap l[p],l[q]
                l = l[:p]+l[q]+l[p+1:q]+l[p]+l[q+1:]
        if mutate&2:
            rep = 1
            while random.randint(0,1): rep += 1
            if random.randint(0,2):
                rep *= 2
            for _ in range(rep):
                p = random.randint(0,len(l)-1)
                # flip l[p]
                l = l[:p]+({'(':')',')':'('}[l[p]])+l[p+1:]

        l = l + '?' + (')' if is_balanced(l) else '(')
        arr = [{'(':1,')':2,'?':3}[c] for c in l]
        arr += [0] * (max_len*2+3-len(arr))
        return arr

    # Generate dataset
    tasks = []
    for i in range(dataset_size):
        tasks.append(gen_single())
    tasks = torch.tensor(tasks) # (dataset_size, max_len*2+3, dataset_size)

    return tasks