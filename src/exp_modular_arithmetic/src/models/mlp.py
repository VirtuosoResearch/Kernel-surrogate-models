import torch
import torch.nn as nn
from collections import OrderedDict

def randint_exclude_zero(p):
    values = torch.arange(-p, p + 1)
    values = values[values != 0]
    return values[torch.randint(0, len(values), (1,))][0]

def modular_addition(x, y, mod):
    """Compute the modular addition of x and y
    """
    return (x + y) % mod

def perturbation_neighborhood(x, p):
    """Compute the perturbation neighborhood of x
    """
    targets = []
    # random sample k from -p to p
    #k = torch.randint(0, 2, (1, x.shape[1]), device=x.device)
    #k = k * 2 - 1
    k = randint_exclude_zero(p)

    target_1 = x.clone()
    target_1[0, :] = modular_addition(target_1[0, :], k, p)
    target_1[2, :] = modular_addition(target_1[2, :], -k, p)
    targets.append(target_1)

    return target_1

class MLP_mnist(nn.Module):
    def __init__(self):
        super(MLP_mnist, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 10)

        self.init_weights = OrderedDict()
        for name, module in self.named_children():
            self.init_weights[name] = module.weight.detach().clone()

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class MLP_arithmetic(nn.Module):
    def __init__(self, dim=128, num_layers=2, num_tokens=97, seq_len=5, activation_name='gelu', dropout_p=0, regression=False):
        super(MLP_arithmetic, self).__init__()
        self.num_tokens = num_tokens
        #self.token_embeddings = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_tokens*2, dim))
        if activation_name == 'relu':
            self.layers.append(nn.ReLU())
        elif activation_name == 'gelu':
            self.layers.append(nn.GELU())
        if dropout_p > 0:
            self.layers.append(nn.Dropout(dropout_p))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(dim, dim))
            if activation_name == 'relu':
                self.layers.append(nn.ReLU())
            elif activation_name == 'gelu':
                self.layers.append(nn.GELU())
            if dropout_p > 0:
                self.layers.append(nn.Dropout(dropout_p))
            
        self.layers.append(nn.Linear(dim, num_tokens))


        self.init_state = self.state_dict()

    def forward(self, x):
        x = torch.eye(self.num_tokens, device=x.device)[x]
        #print(x.shape)
        #x = self.token_embeddings(x)
        x = torch.cat((x[:, 0, :].unsqueeze(1), x[:, 2, :].unsqueeze(1)), dim=-1)
        #x = (x[:, 0, :] + x[:, 2, :]).unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        return x

class MLP_tokenembedding(nn.Module):
    def __init__(self, dim=128, num_layers=2, num_tokens=97, seq_len=5, activation_name='gelu', dropout_p=0, regression=False):
        super(MLP_tokenembedding, self).__init__()
        self.num_tokens = num_tokens
        self.token_embeddings = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(dim*2, dim))
        if activation_name == 'relu':
            self.layers.append(nn.ReLU())
        elif activation_name == 'gelu':
            self.layers.append(nn.GELU())
        if dropout_p > 0:
            self.layers.append(nn.Dropout(dropout_p))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(dim, dim))
            if activation_name == 'relu':
                self.layers.append(nn.ReLU())
            elif activation_name == 'gelu':
                self.layers.append(nn.GELU())
            if dropout_p > 0:
                self.layers.append(nn.Dropout(dropout_p))
            
        self.layers.append(nn.Linear(dim, num_tokens))


        self.init_state = self.state_dict()

    def noisy_embed(self, x, sigma):
        x = x.T
        #print(x)
        target = perturbation_neighborhood(x, self.num_tokens-2)
        #print(target)
        h = self.token_embeddings(x)

        h_target = self.token_embeddings(target)
        direction = (h_target - h).detach()
        unit_vector = direction[[0, 2]] / direction[[0, 2]].norm(dim=-1, keepdim=True)
        unit_random = torch.randn_like(unit_vector)
        #perturabation_vector = unit_vector * unit_random
        #perturabation_vector = unit_random
        perturabation_vector = unit_vector
        
        scale = h[[0,2]].norm() * sigma
        h[[0,2]] += perturabation_vector * scale
        
        return h

    def forward(self, x, embed_noise=False, sigma=0.1):
        if embed_noise:
            x = self.noisy_embed(x, sigma)
        else:
            x = self.token_embeddings(x)
        x = torch.cat((x[:, 0, :].unsqueeze(1), x[:, 2, :].unsqueeze(1)), dim=-1)
        #x = (x[:, 0, :] + x[:, 2, :]).unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        return x