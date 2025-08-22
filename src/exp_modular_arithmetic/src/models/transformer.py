import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from src.models.base import BaseModel

torch.set_printoptions(threshold=10)

activation_map = {
    "relu": nn.ReLU,
    "gelu": nn.GELU
}

def randint_exclude_zero(p):
    values = torch.arange(-p, p + 1)
    values = values[values != 0]
    return values[torch.randint(0, len(values), (1,))][0]

def modular_addition(x, y, mod):
    """Compute the modular addition of x and y
    """
    return (x + y) % mod

def _mod(x, p: int):
    """Positive remainder, works on tensors and scalars."""
    return (x % p + p) % p

def _modinv(x, p):
    return torch.tensor([pow(int(i), -1, p) for i in x])

def mul_transform(x: torch.Tensor, p: int):
    """Transform x by multiplying the first and third rows by a random integer
    """
    a = x[0, :].clone()
    b = x[2, :].clone()

    k = randint_exclude_zero(p-1)

    target = x.clone()
    target[0, :] = _mod(a * k, p)
    k_inv = pow(k.item(), -1, p)
    target[2, :] = _mod(b * k_inv, p)
    mask   = (target[0, :] != a) & (target[2, :] != b) 
    
    return target, mask.bool()


def quad1_transform(x: torch.Tensor, p: int, transform: int | None = None):
    if transform is None:
        transform = torch.randint(0, 12, (), device=x.device).item()

    a = x[0, :].clone()
    b = x[2, :].clone()

    if transform == 0:
        a2, b2 = a, b
    elif transform == 1:
        a2, b2 = a, (-a - b) % p
    elif transform == 2:
        a2, b2 = b, (-a - b) % p
    elif transform == 3:
        a2, b2 = (p - a) % p, (p - b) % p
    elif transform == 4:
        a2, b2 = (p - a) % p, (a + b) % p
    elif transform == 5:
        a2, b2 = (p - b) % p, (a + b) % p
    elif transform == 6:
        a2, b2 = b, a
    elif transform == 7:
        a2, b2 = (-a - b) % p, a
    elif transform == 8:
        a2, b2 = (-a - b) % p, b
    elif transform == 9:
        a2, b2 = (p - b) % p, (p - a) % p
    elif transform == 10:
        a2, b2 = (a + b) % p, (p - a) % p
    elif transform == 11:
        a2, b2 = (a + b) % p, (p - b) % p
    else:
        raise ValueError("transform must be 0-5")

    target = x.clone()
    target[0, :] = _mod(a2, p)
    target[2, :] = _mod(b2, p)
    mask   = (target[0, :] != a) & (target[2, :] != b) 
    
    return target, mask.bool()

def perturbation_neighborhood(x, p):
    """Compute the perturbation neighborhood of x
    """
    targets = []
    # random sample k from -p to p
    #k = torch.randint(0, 2, (1, x.shape[1]), device=x.device)
    #k = k * 2 - 1
    #k = randint_exclude_zero(p-1)
    k = torch.randint(-p + 1, p, (1,x.shape[1])).to(x.device)

    target_1 = x.clone()
    target_1[0, :] = modular_addition(target_1[0, :], k, p)
    target_1[2, :] = modular_addition(target_1[2, :], -k, p)
    targets.append(target_1)

    return target_1

def label_perturbation_neighborhood(x, y, p):
    length, batch_size = x.shape
    target_x = torch.zeros_like(x)
    mask = torch.ones_like(x[0, :])

    equal_matrix = (y.unsqueeze(1) == y.unsqueeze(0)).int().fill_diagonal_(0)

    neighborhood_index = (equal_matrix == 1).int().argmax(dim=1)

    target_x = x.clone()[:, neighborhood_index]

    is_unique = equal_matrix.sum(dim=1) == 0  # shape: [d], bool tensor

    unique_indices = torch.where(is_unique)[0]
    mask[unique_indices] = 0

    return target_x, mask.bool()
    # for i in range(batch_size):
    #     label = y[i]
    #     #print(label)
    #     #print(x[:, i])
    #     same_label_indices = (y == label).nonzero(as_tuple=False).squeeze()
    #     same_label_indices = same_label_indices[same_label_indices != i]
    #     #print(same_label_indices)

    #     if same_label_indices.numel() == 0:
    #         target_x[:, i] = x[:, i]
    #         change_mask[i] = 0
    #         continue
    #     diffs = torch.abs(x[0, i] - x[0, same_label_indices])
    #     valid_mask = diffs > 0
    #     if valid_mask.sum() == 0:
    #         target_x[:, i] = x[:, i]
    #         change_mask[i] = 0
    #     else:
    #         valid_indices = same_label_indices[valid_mask]
    #         valid_dists = diffs[valid_mask]
    #         min_idx = valid_dists.argmin().item()
    #         j = same_label_indices[min_idx]
    #         target_x[:, i] = x[:, j]

    return target_x, change_mask.bool()


class Block(nn.Module):
    """Causal transformer block
    """

    def __init__(self, dim, num_heads, activation_name, dropout_p=0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * num_heads),
            activation_map[activation_name](),
            nn.Linear(dim * num_heads, dim),
        )

    def forward(self, x):
        attn_mask = torch.full(
            (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)
        attn_mask[torch.isnan(attn_mask)] = 0.0 # fixes all 'nan' on 'mps' device

        x = self.ln_1(x)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x


class Decoder(BaseModel):
    """Causal Transformer decoder
    """

    def __init__(self, dim=128, emb_dim=0, num_layers=2, num_heads=4, num_tokens=97, seq_len=5, activation_name='gelu', regression=False, dropout_p=0, task='addition', valid_table=None):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        if emb_dim == 0:
            self.emb_dim = dim
        else:
            self.emb_dim = emb_dim
        self.token_embeddings = nn.Embedding(num_tokens, self.emb_dim)
        self.position_embeddings = nn.Embedding(seq_len, self.emb_dim)
        self.layers = nn.ModuleList()
        if emb_dim > 0:
            self.layers.append(nn.Linear(emb_dim, dim))
        for _ in range(num_layers):
            self.layers.append(Block(dim, num_heads, activation_name, dropout_p))
            if dropout_p > 0 and _ < num_layers - 1:
                self.layers.append(nn.Dropout(dropout_p))

        self.ln_f = nn.LayerNorm(dim)
        #self.batch_norm1 = nn.BatchNorm1d(dim)
        #self.batch_norm2 = nn.BatchNorm1d(dim)
        if not regression:
            self.head = nn.Linear(dim, num_tokens, bias=False)
        else:
            self.head = nn.Linear(dim, 1, bias=False)
        
        self.use_head2 = False

        self.init_state = self.state_dict()
        self.one_hots = torch.eye(num_tokens, num_tokens)

        self.equal_matrix = None
        self.task = task
        self.valid_table = valid_table
        print(self.valid_table)
    
    def get_layers(self):
        layers = OrderedDict()
        if not self.use_head2:
            for name, module in self.named_modules():
                if (type(module) == torch.nn.Linear) and \
                ("LayerNorm" not in name and "embeddings" not in name and "pooler" not in name):
                    layers[name] = module
        #print(layers)
        else:
            for name, module in self.named_modules():
                #if (type(module) == torch.nn.Linear) and 'head' in name:
                #    layers[name] = module
                if (type(module) == torch.nn.Linear) and \
                ("LayerNorm" not in name and "embeddings" not in name and "pooler" not in name):
                    layers[name] = module
            layers.pop('head')
        return layers

    def build_equal_matrix(self, a, b, y):
        target_x = torch.zeros_like(a)
        mask = torch.ones_like(a)

        equal_matrix = (y.unsqueeze(1) == y.unsqueeze(0)).int().fill_diagonal_(0)

        neighborhood_index = (equal_matrix == 1).int().argmax(dim=1)

        target_a = a.clone()[:, neighborhood_index]
        target_b = b.clone()[:, neighborhood_index]

        is_unique = equal_matrix.sum(dim=1) == 0  # shape: [d], bool tensor

        unique_indices = torch.where(is_unique)[0]
        mask[unique_indices] = 0


    def noisy_embed(self, x, y, sigma, method='label_noise'):
        x = x.T
        y = y.T
        #print(x)
        #target = perturbation_neighborhood(x, self.num_tokens-2)
        if method == 'rule_noise':
            if self.task == 'addition':
                target = perturbation_neighborhood(x, self.num_tokens-2)
            elif self.task == 'mul':
                target, mask = mul_transform(x, self.num_tokens-2)
            elif self.task == 'quad1':
                target, mask = quad1_transform(x, self.num_tokens-2)
        #mask = torch.ones_like(x[0, :]).bool()
        elif method == 'label_noise':
            target, mask = label_perturbation_neighborhood(x, y, self.num_tokens-2)
        #print(target)
        mask = torch.ones_like(x[0, :]).bool()
        h = self.token_embeddings(x)

        if self.valid_table is not None:
            self.valid_table = self.valid_table.to(x.device)
            selected_values = self.valid_table[target[0, :], target[2, :]]
            valid_mask = (selected_values == 1)
            target[:, valid_mask] = x[:, valid_mask]

        target = target.detach()
        h_target = self.token_embeddings(target)
        
        direction = (h_target - h).detach()
        #unit_vector = direction[[0,2]] / (direction[[0,2]].norm(dim=-1, keepdim=True) + 1e-9)
        #unit_random = torch.randn_like(unit_vector)
        #perturbation_vector = unit_random
        #perturbation_vector = unit_vector
        perturbation_vector = direction
        
        #scale = h[[0,2]].norm(dim=-1, keepdim=True) * sigma
        scale = sigma
        #h[[0,2]] += perturbation_vector * scale
        #h[0] += perturbation_vector[0] * scale
        #h[2] += perturbation_vector[1] * scale
        #print(h[0, :])
        positions = torch.arange(x.shape[0], device=x.device).unsqueeze(-1)
        if sigma == 1:
            h = h_target
        else:
            h[0, mask] += perturbation_vector[0, mask] * scale
            h[2, mask] += perturbation_vector[2, mask] * scale
        h = h + self.position_embeddings(positions).expand_as(h)
        
        return h

    def embed(self, x):
        x = x.T
        h = self.token_embeddings(x)
        positions = torch.arange(x.shape[0], device=x.device).unsqueeze(-1)
        h = h + self.position_embeddings(positions).expand_as(h)
        return h

    def forward_(self, x, y=None, embed_noise=False, sigma=0.1, method=None):
        if embed_noise:
            h = self.noisy_embed(x, y, sigma, method)
        else:
            h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
            #src_perm = h.permute(1, 2, 0)  # (seq, batch, features) -> (batch, features, seq)
            #src_norm = self.batch_norm1(src_perm)
            #h = src_norm.permute(2, 0, 1)  # Back to (seq, batch, features)

        h = self.ln_f(h)
        logits = self.head(h)

        logits = logits.permute(1, 0, 2)
        return logits
    
    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
            #src_perm = h.permute(1, 2, 0)  # (seq, batch, features) -> (batch, features, seq)
            #src_norm = self.batch_norm1(src_perm)
            #h = src_norm.permute(2, 0, 1)  # Back to (seq, batch, features)
        h = self.ln_f(h)
        logits = self.head(h)

        logits = logits.permute(1, 0, 2)
        return logits

    def get_rep(self, x):
        x = x.T
        h = self.token_embeddings(x)
        positions = torch.arange(x.shape[0], device=x.device).unsqueeze(-1)
        h = h + self.position_embeddings(positions).expand_as(h)
        for layer in self.layers:
            h = layer(h)
            #src_perm = h.permute(1, 2, 0)  # (seq, batch, features) -> (batch, features, seq)
            #src_norm = self.batch_norm1(src_perm)
            #h = src_norm.permute(2, 0, 1)  # Back to (seq, batch, features)

        h = self.ln_f(h)

        return h.permute(1, 0, 2)
    
    def add_noise_forward(self, x, mean=0, std=0.1):
        self.eval()
        x = x.T
        h = self.token_embeddings(x)
        h_original = h.clone()
        noise = torch.normal(mean, std, size=h.size()).to(h.device)
        #print(noise.shape)
        #print(h.shape)
        h[0] += noise[0]
        h[2] += noise[2]
        noise = noise.permute(1, 0, 2)
        noise_norm = noise.norm(dim=-1)[:, 0] + noise.norm(dim=-1)[:, 2]
        #print(noise_norm.mean())
        positions = torch.arange(x.shape[0], device=x.device).unsqueeze(-1)
        h = h + self.position_embeddings(positions).expand_as(h)
        for layer in self.layers:
            h = layer(h)
            #src_perm = h.permute(1, 2, 0)  # (seq, batch, features) -> (batch, features, seq)
            #src_norm = self.batch_norm1(src_perm)
            #h = src_norm.permute(2, 0, 1)  # Back to (seq, batch, features)

        h = self.ln_f(h)
        logits = self.head(h)
        # TODO JL 10/26/24: get rid of the hacky transpose
        logits = logits.permute(1, 0, 2)
        self.train()
        return logits, noise_norm
    
    def add_bi_noise_forward(self, x, mean=0, std=0.1):
        self.eval()
        x = x.T
        h_1 = self.token_embeddings(x)
        h_2 = self.token_embeddings(x)
        noise = torch.normal(mean, std, size=h_1.size()).to(h_1.device)
        #print(noise.shape)
        h_1[0] += noise[0]
        h_1[2] += noise[2]
        h_2[0] -= noise[0]
        h_2[2] -= noise[2]
        noise = noise.permute(1, 0, 2)
        noise_norm = noise.norm(dim=-1)[:, 0] + noise.norm(dim=-1)[:, 2]
        #print(noise_norm.mean())
        positions = torch.arange(x.shape[0], device=x.device).unsqueeze(-1)
        h_1 = h_1 + self.position_embeddings(positions).expand_as(h_1)
        h_2 = h_2 + self.position_embeddings(positions).expand_as(h_2)
        for layer in self.layers:
            h_2 = layer(h_2)
            h_1 = layer(h_1)
            #src_perm = h.permute(1, 2, 0)  # (seq, batch, features) -> (batch, features, seq)
            #src_norm = self.batch_norm1(src_perm)
            #h = src_norm.permute(2, 0, 1)  # Back to (seq, batch, features)

        h_1 = self.ln_f(h_1)
        h_2 = self.ln_f(h_2)
        logits_1 = self.head(h_1)
        logits_2 = self.head(h_2)
        logits_1 = logits_1.permute(1, 0, 2)
        logits_2 = logits_2.permute(1, 0, 2)
        self.train()
        return logits_1, logits_2, noise_norm
    
    def save_init_state(self):
        self.init_state = self.state_dict()
    
    def get_init_W(self):
        return self.W0
    
    def get_W(self):
        return self.head.weight.detach().clone()

    def reset_head(self):
        #for layer in self.layers:
        #    for param in layer.parameters():
        #        param.requires_grad = False
        #self.head = nn.Linear(self.dim, self.num_tokens, bias=False)
        for param in self.head.parameters():
            param.requires_grad = False
        
        self.use_head2 = True

class embed_FNN(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim):
        super(embed_FNN, self).__init__()
        self.fc1 = nn.Embedding(input_dim, embed_dim)    # first layer: input_dim → 4
        self.fc2 = nn.Linear(embed_dim*2, embed_dim)   # second layer: 4 → output_dim
        self.fc3 = nn.Linear(embed_dim, output_dim)    # third layer: output_dim → 1

    def forward(self, x):
        h = self.fc1(x)                       # embedding layer
        h = h.permute(1, 0, 2)              # (batch_size, seq_len, embed_dim) → (seq_len, batch_size, embed_dim)
        h = torch.cat((h[:, 0, :].unsqueeze(1), h[:, 2, :].unsqueeze(1)), dim=-1)
        h = F.relu(h)               # activation after second layer
        h = self.fc2(h)                       # output layer (no activation if regression; softmax if classification)
        h = F.relu(h)               # activation after second layer
        h = self.fc3(h)                       # output layer (no activation if regression; softmax if classification)

        return h

class Grok_Transfer(BaseModel):
    def __init__(self, transfer_dim=16, dim=128, num_layers=2, num_heads=4, num_tokens=97, seq_len=5, activation_name='gelu', regression=False, dropout_p=0):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens

        self.transfer_embedding = embed_FNN(num_tokens, transfer_dim, num_tokens)
        self.transfer_adapter = nn.Linear(transfer_dim, dim)
        self.transfer_head = nn.Linear(transfer_dim*2, num_tokens)

        self.position_embeddings = nn.Embedding(seq_len, dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Block(dim, num_heads, activation_name))

        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_tokens, bias=False)

    def forward(self, x, train_transfer=False):
        x = x.T
        if train_transfer:
            h = self.transfer_embedding(x)

            return h
        else:
            h = self.transfer_embedding.fc1(x)
            h = self.transfer_adapter(h)
            positions = torch.arange(x.shape[0], device=x.device).unsqueeze(-1)
            h = h + self.position_embeddings(positions).expand_as(h)
            for layer in self.layers:
                h = layer(h)

            h = self.ln_f(h)
            logits = self.head(h)

            logits = logits.permute(1, 0, 2)
            return logits