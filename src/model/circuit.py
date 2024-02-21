import argparse
import math

import numpy as np
from pysat.formula import CNF
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import gc
from .pgates import *


class BaseCircuit(nn.Module):
    """Base class for all circuits"""

    def __init__(self, **kwargs):
        """
        Args:
            input_shape (list(tuple(int, int))): list of tuples, each tuple representing the shape of a primary input
            use_pgates (bool, optional): whether to use pgates or not. Defaults to True.
            device (str, optional): device to use. Defaults to 'cuda'.
            batch_size (int, optional): batch size. Defaults to 1.
        """
        super().__init__()
        self.cnf_problem = kwargs["cnf_problem"]
        self.clause_list = self.cnf_problem.clauses

        max_clause_len = max([len(clause) for clause in self.clause_list])
        flat_var_list = np.array([clause + [0] * (max_clause_len - len(clause)) for clause in self.clause_list])

        total_variables = []
        while True:
            variables = flat_var_list[np.expand_dims((np.absolute(flat_var_list) > 0).sum(axis = -1)==1, axis = 1) & (np.absolute(flat_var_list) > 0) ]
            if len(variables) > 0:
                total_variables.append(variables)
                flat_var_list = np.delete(flat_var_list, np.argwhere(np.isin(flat_var_list, variables).sum(axis = -1) > 0),0)
                flat_var_list[np.isin(flat_var_list, -1 * variables)] = 0
            else:
                break

        self.total_variables = np.concatenate(total_variables,axis = 0)
        self.max_clause_len = max([len(clause) for clause in self.clause_list])
        self.flat_var_list = torch.IntTensor(flat_var_list)
        
        self.index = torch.IntTensor(self.total_variables)
        print(self.index, self.index.shape)
        self.input_embedding = PIEmbedding(
            index = self.index,
            input_shape=kwargs["input_shape"],
            use_pgates=kwargs["use_pgates"],
            device=kwargs["device"],
            batch_size=kwargs["batch_size"],
        )
        self.device = kwargs["device"]
        self.layers = {}

    def build_model(self, **kwargs):
        raise NotImplementedError

    def forward(self, input):
        raise NotImplementedError


class PIEmbedding(nn.Module):
    def __init__(
        self,
        index: torch.int,
        input_shape: list[int],
        use_pgates: bool = True,
        device: str = "cpu",
        batch_size: int = 1,
    ):
        """Embedding Layers for Primary Inputs

        Args:
            input_shape (list(int)):
                list of ints where each int represents the bitwidth of a primary input
                e.g. for SAT problems with k variables, input_shape = [k]
                    for a two-input 8-bit adder, input_shape = [8, 8]
            use_pgates (bool, optional): whether to use pgates or not. Defaults to True.
            device (str, optional): device to use. Defaults to 'cuda'.
            batch_size (int, optional): batch size. Defaults to 1.
        """
        super().__init__()
        self.input_shape = input_shape
        self.use_pgates = use_pgates
        self.device = device
        self.batch_size = batch_size
        self.index = index

        
        
        '''init_value = torch.randn(input_shape[0], self.batch_size  )
        init_value[self.index.abs() - 1,:] = ((self.index > 0) * 7. - 3.5).unsqueeze(-1)
        init_value = init_value.to(device)'''

        # self.embeddings holds a list of nn.Embedding objects
        # each nn.Embedding representing a bit-vector of a primary input
        self.embeddings = torch.nn.parameter.Parameter( torch.randn(input_shape[0], self.batch_size, device = device  ) )

                
        self.activation = torch.nn.Sigmoid()  # if use_pgates else Sgn()


    def forward(self, input: torch.Tensor):
        self.embeddings.data.add_(0.25 * torch.randn(self.input_shape[0], self.batch_size).to(self.embeddings.data.device))
        self.embeddings.data.clamp_(-3.5, 3.5)
        x = self.activation( 2 * self.embeddings ) 
        
        x = torch.concat((torch.ones(1, x.shape[1]).to(self.device), 1. - x, x), dim=0)
        
        x = torch.nn.functional.embedding(input, x)
        return x

    def get_weights(self):
        """Get weights of the embedding layers"""
        weights = self.embeddings.data.permute(1,0)
        weights[:,self.index.abs() - 1] = ((self.index > 0) * 7. - 3.5).unsqueeze(0).to(weights.device)
        return weights


class CNF2Circuit(BaseCircuit):
    """Combinational Circuit instantiated from a PySAT CNF problem"""

    def __init__(self, **kwargs):
        # read cnf file
        assert kwargs["cnf_problem"] is not None
        self.cnf_problem = kwargs["cnf_problem"]
        self.use_pgates = kwargs["use_pgates"]
        
        # define input shape
        # for SAT problems, input is a single k-bit vector
        # where k is the number of variables in the problem
        self.input_shape = [self.cnf_problem.nv]

        # generate input embedding layer
        super().__init__(input_shape=self.input_shape, **kwargs)
        

        self.emb =self.input_embedding.to(self.device)

        
        self.input = torch.nn.parameter.Parameter(self.flat_var_list, requires_grad = False).to(self.device)#[0:900000*self.max_clause_len]
        
        
        self.input = torch.where(self.input > -1, torch.abs(self.input), torch.abs(self.input) + self.cnf_problem.nv)
        
        

    def forward(self):
        # batchsize is x.shape[0]
        x = self.emb(self.input)
        x = x.permute(2, 0, 1)
        y = x
        #x = 1 - torch.prod(1 - x, dim = -1).permute(1, 0)
        x = torch.prod(x, dim = -1).permute(1, 0)
        return x

    def get_input_weights(self, idx=0):
        """Get weights of the input embedding layer"""
        '''assert self.layers["emb"] is not None
        weights = self.layers["emb"].get_weights()[0][idx]'''
        weights = torch.sign(self.emb.get_weights())
        if self.use_pgates:
            return ((weights + 1.0) / 2.0)
        else:
            raise NotImplementedError
