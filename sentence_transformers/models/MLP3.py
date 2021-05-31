import torch
from torch import nn
import os
import json
from typing import Union, Tuple, List, Iterable, Dict
from torch import Tensor

class MLP3(nn.Module):
    def __init__(self, hidden_dim=2048, norm=None, activation='relu'):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out-
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d.
        This MLP has 3 layers.
        '''
        self.config_keys = ['hidden_dim',  'norm', 'activation']
        self.hidden_dim = hidden_dim
        self.norm = norm
        self.activation = activation
        
        if activation == "relu":
            activation_layer = nn.ReLU()
        elif activation == "leakyrelu":
            activation_layer = nn.LeakyReLU()
        elif activation == "tanh":
            activation_layer = nn.Tanh()
        elif activation == "sigmoid":
            activation_layer = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation function {hidden_activation}")
            
        if norm:
            if norm=='bn':
                norm_layer = nn.BatchNorm1d
            else: 
                norm_layer = nn.LayerNorm

            self.layer1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                norm_layer(hidden_dim),
                nn.ReLU(inplace=True)
            )
            self.layer2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                norm_layer(hidden_dim),
                nn.ReLU(inplace=True)
            )
            self.layer3 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                norm_layer(hidden_dim)
            )
        else:
            self.layer1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True)
            )
            self.layer2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True)
            )
            self.layer3 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
            )
           
        self.num_layers = 3

    def set_layers(self, num_layers):
        self.num_layers = num_layers
 
    def forward(self, features: Dict[str, Tensor]):
        x = features["token_embeddings"]
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        features["token_embeddings"] = x
        return features
    
    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}
    
    def save(self, output_path):
        with open(os.path.join(output_path, 'mlp3_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'mlp3_config.json')) as fIn:
            config = json.load(fIn)

        return MLP3(**config)