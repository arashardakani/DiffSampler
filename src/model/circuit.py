import argparse
import math

import numpy as np
from pysat.formula import CNF
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

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
        self.input_embedding = PIEmbedding(
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
        # self.embeddings holds a list of nn.Embedding objects
        # each nn.Embedding representing a bit-vector of a primary input
        self.embeddings = nn.ModuleList(
            [nn.Embedding(self.batch_size, input_width) for input_width in input_shape]
        )
        self.activation = torch.nn.Sigmoid()  # if use_pgates else Sgn()

        # initialize embeddings
        for emb in self.embeddings:
            emb.weight.data = torch.rand(emb.weight.data.shape)

    def forward(self, input: torch.Tensor):
        x = torch.cat([emb(input) for emb in self.embeddings], dim=1)
        out = self.activation(x)
        return out

    def get_weights(self):
        """Get weights of the embedding layers"""
        weights = [emb.weight.data for emb in self.embeddings]
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

        self.clause_list = self.cnf_problem.clauses
        self.max_clause_len = max([len(clause) for clause in self.clause_list])
        self.flat_var_list = (
            np.array(
                [
                    clause + [0] * (self.max_clause_len - len(clause))
                    for clause in self.clause_list
                ]
            )
            .flatten()
            .tolist()
        )
        self.var_tensor = torch.LongTensor(self.flat_var_list).to(self.device)
        self.var_negation_tensor = torch.LongTensor(
            [0 if v >= 0 else 1 for v in self.flat_var_list]
        ).to(self.device)
        # package all layers into a dictionary
        self.layers = {
            "emb": self.input_embedding,
        }

    def forward(self, input):
        # batchsize is x.shape[0]
        x = self.layers["emb"](input)
        x = torch.concat((torch.zeros(x.shape[0], 1).to(x.device), x), dim=1)
        gather_x = torch.index_select(x, -1, torch.abs(self.var_tensor))
        gather_x = torch.concat(
            (gather_x.unsqueeze(-1), 1 - gather_x.unsqueeze(-1)), dim=-1
        )
        gather_x = torch.gather(
            gather_x,
            x.dim(),
            self.var_negation_tensor.repeat((x.shape[0], 1)).unsqueeze(-1),
        ).squeeze(-1)
        reshaped_x = torch.reshape(gather_x, (x.shape[0], -1, self.max_clause_len))
        output = 1 - torch.prod(1 - reshaped_x, dim=-1)
        return output

    def get_input_weights(self, idx=0):
        """Get weights of the input embedding layer"""
        assert self.layers["emb"] is not None
        weights = self.layers["emb"].get_weights()[0][idx]
        if self.use_pgates:
            return ((torch.sign(weights) + 1.0) / 2.0).long().cpu().tolist()
        else:
            raise NotImplementedError
