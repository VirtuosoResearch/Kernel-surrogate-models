import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict



class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.init_state = self.state_dict()
    
    def get_layers(self):
        layers = OrderedDict()
        for name, module in self.named_modules():
            if (type(module) == torch.nn.Linear) and \
            ("LayerNorm" not in name and "embeddings" not in name and "pooler" not in name):
                layers[name] = module
        return layers
    
