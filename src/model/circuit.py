import argparse
import math

import numpy as np
from pysat.formula import CNF
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from utils_gates import *

class BaseCircuit(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_embedding = PIEmbedding(
            input_shape=kwargs['input_shape'],
            use_pgates=kwargs['use_pgates'],
            device=self.device
        )
        self.layers = []

    def forward(self, input):
        raise NotImplementedError


class PIEmbedding(nn.Module):
    def __init__(self, input_shape: list(tuple(int, int)), use_pgates:bool=True, device:str='cuda'):
        """Embedding Layers for Primary Inputs

        Args:
            input_shape (list(tuple(int, int))): list of tuples, each tuple representing the shape of a primary input
            use_pgates (bool, optional): whether to use pgates or not. Defaults to True.
            device (str, optional): device to use. Defaults to 'cuda'.

        
        """
        super().__init__()
        self.input_shape = input_shape
        self.use_pgates = use_pgates
        self.device = device
        # self.data holds a list of nn.Embedding objects
        # each nn.Embedding representing a bit-vector of a primary input
        self.data = []
        for primary_input in input_shape:
            p_in = nn.Embedding(primary_input[0], primary_input[1])
            self.data.append(p_in)
        self.data = nn.ModuleList(self.data)
        self.activation = torch.nn.Sigmoid() if use_pagtes else Sgn()

    def forward(self, input: torch.Tensor):
        """Forward pass of PIEmbedding
        
        Args:
            input (torch.Tensor): input tensor of shape (batch_size, num_primary_inputs)
        """
        data = []
        # concatenate input vectors to form a single tensor
        for i in range(len(self.input_shape)):
            data.append(self.data[i](input[:,i]))
        data = torch.cat(data, dim=1)
        out = self.activation(data)
        return out
    


