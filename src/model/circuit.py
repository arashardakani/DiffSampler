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

        '''var_list = []
        stop = False
        while not stop:
            init_len = len(self.clause_list)
            i = 0
            for clause in self.clause_list:
                if len(clause) == 1:
                    var_list.append(clause[0])
                    self.clause_list.remove(self.clause_list[i])
                    break
                i += 1
            if len(self.clause_list) == init_len:
                stop = True
            i = 0
            for clause in self.clause_list:
                if var_list[-1] in clause:
                    self.clause_list.remove(self.clause_list[i])
                    break
                elif -var_list[-1] in clause:
                        self.clause_list[i].remove(-var_list[-1])
                i += 1'''
        
        self.max_clause_len = max([len(clause) for clause in self.clause_list])
        self.flat_var_list = np.array([clause + [0] * (self.max_clause_len - len(clause)) for clause in self.clause_list]).tolist() #.flatten()
        
        self.index = torch.IntTensor(self.flat_var_list)

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

        # self.embeddings holds a list of nn.Embedding objects
        # each nn.Embedding representing a bit-vector of a primary input
        '''self.embeddings = nn.ModuleList(
            [nn.Embedding(input_width, self.batch_size) for input_width in input_shape]
        )''' #torch.randn(input_shape[0], self.batch_size)
        initial_index = index
        indices_total = initial_index[torch.argwhere((initial_index != 0).sum(dim=-1)==1)][:,0,0]
        indices = indices_total
        '''while (True):
            ##indices = initial_index[torch.argwhere((initial_index != 0).sum(dim=-1)==1)][:,0,0]
            condition = ((indices.view(-1, 1) - torch.abs(initial_index).view(-1)).transpose(-1, -2) == 0).sum(dim=-1).view(initial_index.shape) != 0
            if condition.sum() == 0:
                break
            initial_index = torch.where(condition, 0, initial_index)
            indices = initial_index[torch.argwhere((initial_index != 0).sum(dim=-1)==1)][:,0,0]
            indices_total = torch.concat((indices_total, indices), dim = -1)'''

        init_value = torch.randn(input_shape[0], self.batch_size, device = device  )
        self.embeddings = torch.nn.parameter.Parameter(torch.randn(input_shape[0], self.batch_size, device = device  )) #nn.Embedding( self.batch_size, input_shape[0]) #

        ##self.embeddings = torch.nn.parameter.Parameter(torch.load('./emb_weight1.npy'))
        self.mask = torch.abs(self.embeddings) < 10.
        #self.embeddings = torch.nn.parameter.Parameter(torch.load('sol.npy').unsqueeze(1)) 
        self.activation = torch.nn.Sigmoid()  # if use_pgates else Sgn()
        self.dropout = nn.Dropout(p=0.05)

        # initialize embeddings
        '''for emb in self.embeddings:
            emb.weight.data = torch.rand(emb.weight.data.shape)'''

    def forward(self, input: torch.Tensor, cond, true_index, comp_index):
        #x = (input) #torch.cat([emb(input) for emb in self.embeddings], dim=1)
        #print(self.embeddings.weight.data)
        #x = self.activation(self.embeddings(torch.IntTensor([0]).to(self.device)).permute(1,0) )
        #print(torch.cuda.memory_reserved(0))
        #print(torch.cuda.memory_allocated(0))
        self.embeddings.data.clamp_(-7.5, 7.5)
        '''if cond:
            self.embeddings.data = torch.where((self.embeddings > -0.5) * (self.embeddings < 0.5), 3* torch.sign(self.embeddings), self.embeddings)'''
        x = sampler(self.embeddings, self.mask)
        ##x = self.embeddings
        
        #x = wherefunc(x)
        ##x = sigmfunc(x)
        x = self.activation( x ) 
        #x = torch.clamp(x, 0., 1.)
        #x = wherefunc(x)
        #x = torch.nn.functional.sigmoid(wherefunc(self.embeddings))
        #x = wherefunc(x)
        #x = torch.clamp(self.embeddings, min=0., max=1.)
        
        #print(torch.cuda.memory_reserved(0))
        #print(torch.cuda.memory_allocated(0))
        
        ####x = torch.concat((torch.zeros(1, x.shape[1]).to(self.device), x, 1. - x), dim=0)
        

        x = wherefunc(x)

        x = torch.concat((torch.zeros(1, x.shape[1]).to(self.device), x, 1. - x), dim=0)
        #print(torch.cuda.memory_reserved(0))
        #print(torch.cuda.memory_allocated(0))
        x = blockgrad(x, true_index, comp_index)
        x = torch.nn.functional.embedding(input, x)
        #print(torch.cuda.memory_reserved(0))
        #print(torch.cuda.memory_allocated(0))
        #weights = wherefunc(weights)
        return x

    def get_weights(self):
        """Get weights of the embedding layers"""
        #weights = [emb.weight.data[1:,:].permute(1,0) for emb in self.embeddings]
        #weights = self.embeddings.weight.data[1:,:].permute(1,0)
        weights = self.embeddings.data.permute(1,0)
        return weights

'''class OR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, var):
        #x  =torch.where(var, 1 - input, input)
        x = 1 - torch.prod(1-input, dim=-1)
        return x'''

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
        '''self.clause_list = self.cnf_problem.clauses
        self.max_clause_len = max([len(clause) for clause in self.clause_list])
        self.flat_var_list = np.array([clause + [0] * (self.max_clause_len - len(clause)) for clause in self.clause_list]).tolist() #.flatten()
        self.var_tensor = torch.IntTensor(self.flat_var_list).to(self.device)#[0:900000*self.max_clause_len]
        self.var_negation_tensor = (self.var_tensor < 0.).unsqueeze(0)
        self.var_tensor.abs_()'''
        '''self.layers = {
            "emb": self.input_embedding,
        }'''

        
        '''self.clause_list = self.cnf_problem.clauses
        
        self.max_clause_len = max([len(clause) for clause in self.clause_list])
        self.flat_var_list = np.array([clause + [0] * (self.max_clause_len - len(clause)) for clause in self.clause_list]).tolist() #.flatten()
        
        self.index = torch.IntTensor(self.flat_var_list)'''
        #self.index[torch.argwhere((self.index != 0).sum(dim=-1)==1)] = 0

        self.emb =self.input_embedding.to(self.device)

        self.index_comp = torch.bincount(torch.where(self.index.reshape(-1) > 0, 0, self.index.reshape(-1))* -1, minlength = self.input_shape[0] + 1).to(self.device)
        self.index_true = torch.bincount(torch.where(self.index.reshape(-1) < 0, 0, self.index.reshape(-1)), minlength = self.input_shape[0] + 1).to(self.device)
        #print(self.index_true)
        #print(self.index_comp)
        self.input = torch.nn.parameter.Parameter(torch.IntTensor(self.flat_var_list),requires_grad = False).to(self.device)#[0:900000*self.max_clause_len]
        #print(self.input[6])
        ##self.input = torch.where(self.input < 1, torch.abs(self.input), torch.abs(self.input) + self.cnf_problem.nv)
        self.input = torch.where(self.input > -1, torch.abs(self.input), torch.abs(self.input) + self.cnf_problem.nv)
        #print(self.input[6])
        #print(self.input.shape)
        self.in_ = self.input.clone()
        ###self.input = torch.where(self.input < 0, torch.abs(self.input) + self.cnf_problem.nv, self.input) original
        '''print(torch.cuda.memory_reserved(0))
        print(torch.cuda.memory_allocated(0))
        torch.cuda.empty_cache()
        gc.collect()
        print((self.input == 0).sum(), self.input.shape)
        print(torch.cuda.memory_reserved(0))
        print(torch.cuda.memory_allocated(0))'''
         #nn.Embedding(self.cnf_problem.nv + 1, 1) 
        #self.OR = torch.nn.DataParallel(OR(), device_ids=[0])
    
    '''def operations(self, input):
        """performing negation and OR operations"""
        return ((torch.sign(weights) + 1.0) / 2.0).long().cpu().tolist()
    '''  

    def forward(self, idx1, idx2, cond):
        # batchsize is x.shape[0]
        #print('inside forward, x.device', input.device, input.shape, mask.device, mask.shape)
        #print('inside forward, x.device {}'.format(mask.device))
        #self.layers["emb"].embeddings[0].weight.data[0,:].mul_(0.).add_(-100000.)
        #print(self.layers["emb"].embeddings[0].weight.device)
        #x = self.layers["emb"](input).permute(2, 0, 1)
        #self.emb.embeddings.weight.data[0,:].mul_(0.).add_(-100000.)
        #print(self.layers["emb"].embeddings[0].weight.device)
        '''print(self.input[6])
        print(self.in_[6])
        print(self.input.shape)'''
        #print(self.in_[330])
        x = self.emb(self.in_[idx1:idx2, :], cond, self.index_true, self.index_comp).permute(2, 0, 1)
        
        #x = wherefunc(x)
        #print(torch.cuda.memory_reserved(0))
        #print(torch.cuda.memory_allocated(0))
        '''x = torch.concat((torch.zeros(x.shape[0], 1).to(x.device), x), dim=1)
        x = torch.index_select(x, -1, self.var_tensor).reshape(-1, self.max_clause_len)'''
        #x = torch.where(self.mask[idx1:idx2, :].unsqueeze(0), 1 - x, x)
        #print(torch.cuda.memory_reserved(0))
        #print(torch.cuda.memory_allocated(0))
        #########x = 1 - torch.prod(1-x, dim=-1).permute(1, 0) #self.OR(x, self.var_negation_tensor) # original

        y = x
        #print(x[0,330])
        ##x = torch.prod(1- x, dim=-1).permute(1, 0)
        x = 1 - prod(1 - x).permute(1, 0)
        ##x = 1 - torch.sum(torch.log2(1-x), dim=-1).permute(1, 0)
        '''if cond:
            x = wherefunc(x)'''
        return x, x, y, self.in_[idx1:idx2, :]#, torch.sum(torch.log2(x), dim = 0) #torch.prod(x, dim=0)

    def get_input_weights(self, idx=0):
        """Get weights of the input embedding layer"""
        '''assert self.layers["emb"] is not None
        weights = self.layers["emb"].get_weights()[0][idx]'''
        weights = self.emb.get_weights()[idx]
        if self.use_pgates:
            return ((torch.sign(weights) + 1.0) / 2.0).long().cpu().tolist()
        else:
            raise NotImplementedError



class where(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        x = torch.where((x > 0.499) & (x < 0.501), 0.51, x)
        #x = torch.where(x == 0.5, 0.51, x)
        #x = torch.bernoulli(x)
        #x = torch.where((x < 0.75) * (x > 0.25), torch.clamp(torch.bernoulli(x), 0.25, 0.75), x)
        ##x = torch.where(x<1., 0.01 * x, x)
        #x = x * 0.01
        #x = (torch.sign(x-0.5)+1.)/2.
        
        return x


    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        #print(grad_output, grad_output.shape)
        #print(grad_output, grad_output.max(), grad_output.min())
        #print(torch.abs(grad_output).max(),'here***********************')
        #print(grad_output.permute(2,1,0)[2,14],grad_output.permute(2,1,0)[1,15],grad_output.permute(2,1,0)[2,24],grad_output.permute(2,1,0)[1,25],grad_output.permute(2,1,0)[2,36],grad_output.permute(2,1,0)[1,37],grad_output.permute(2,1,0)[1,46],grad_output.permute(2,1,0)[2,47],grad_output.permute(2,1,0)[2,69],grad_output.permute(2,1,0)[2,70],grad_output.permute(2,1,0)[2,77],grad_output.permute(2,1,0)[2,78],grad_output.permute(2,1,0)[2,87],grad_output.permute(2,1,0)[2,88],grad_output.permute(2,1,0)[2,95],grad_output.permute(2,1,0)[2,96],grad_output.permute(2,1,0)[0,97])
        #print('******************************************************************')
        return grad_output


wherefunc = where.apply



class samplerfunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor):
        ##print(y.sum(), y.shape)
        ctx.save_for_backward(y, x)
        #x = torch.bernoulli(x)
        #x = (torch.sign(x-0.5)+1.)/2.
        # x = torch.where((x > -1.5) * (x < 1.5), torch.sign(x) * 1.5, x)
        return x
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        ##print(torch.abs(grad_output).max())
        #sdads
        #print(torch.abs(grad_output).max(),'here***********************')
        #print(grad_output.permute(2,1,0)[2,14],grad_output.permute(2,1,0)[1,15],grad_output.permute(2,1,0)[2,24],grad_output.permute(2,1,0)[1,25],grad_output.permute(2,1,0)[2,36],grad_output.permute(2,1,0)[1,37],grad_output.permute(2,1,0)[1,46],grad_output.permute(2,1,0)[2,47],grad_output.permute(2,1,0)[2,69],grad_output.permute(2,1,0)[2,70],grad_output.permute(2,1,0)[2,77],grad_output.permute(2,1,0)[2,78],grad_output.permute(2,1,0)[2,87],grad_output.permute(2,1,0)[2,88],grad_output.permute(2,1,0)[2,95],grad_output.permute(2,1,0)[2,96],grad_output.permute(2,1,0)[0,97])
        #print('******************************************************************')
        return grad_output , None #* ctx.saved_tensors[0]


sampler = samplerfunc.apply



class sigm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        #x = torch.bernoulli(x)
        #x = (torch.sign(x-0.5)+1.)/2.
        #x = torch.where((x > -0.5) * (x < 0.5), torch.sign(x) * 1.5, x)
        return torch.nn.functional.sigmoid(x)


    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        #print(torch.abs(grad_output).max(),'here***********************')
        #print(grad_output.permute(2,1,0)[2,14],grad_output.permute(2,1,0)[1,15],grad_output.permute(2,1,0)[2,24],grad_output.permute(2,1,0)[1,25],grad_output.permute(2,1,0)[2,36],grad_output.permute(2,1,0)[1,37],grad_output.permute(2,1,0)[1,46],grad_output.permute(2,1,0)[2,47],grad_output.permute(2,1,0)[2,69],grad_output.permute(2,1,0)[2,70],grad_output.permute(2,1,0)[2,77],grad_output.permute(2,1,0)[2,78],grad_output.permute(2,1,0)[2,87],grad_output.permute(2,1,0)[2,88],grad_output.permute(2,1,0)[2,95],grad_output.permute(2,1,0)[2,96],grad_output.permute(2,1,0)[0,97])
        #print('******************************************************************')
        return grad_output


sigmfunc = sigm.apply


class blockgradfunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, ture_index, comp_index):
        ctx.save_for_backward(ture_index, comp_index)
        #x = torch.bernoulli(x)
        #x = (torch.sign(x-0.5)+1.)/2.
        #x = torch.where((x > -0.5) * (x < 0.5), torch.sign(x) * 1.5, x)
        return x


    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        true_index, comp_index = ctx.saved_tensors
        weights = torch.concat((torch.ones(1, grad_output.shape[1]).to(grad_output.device), true_index[1:].unsqueeze(-1), comp_index[1:].unsqueeze(-1)), dim=0)
        weights = torch.where(weights == 0, 1, weights)
        ##print(grad_output, grad_output/weights)
        ##print(torch.abs(grad_output/weights)[1:].max(), torch.abs(grad_output)[1:].max())
        ##print( ((grad_output/weights)[1:551]-(grad_output/weights)[551:])[0:10] )
        #print( (torch.abs(grad_output/weights)[1:551]-torch.abs(grad_output/weights)[551:])[0:10] )
        '''print(grad_output)
        print(grad_output/weights)'''
        ###print(grad_output/weights, grad_output, weights)
        #print(torch.abs(grad_output).max(),'here***********************')
        #print(grad_output.permute(2,1,0)[2,14],grad_output.permute(2,1,0)[1,15],grad_output.permute(2,1,0)[2,24],grad_output.permute(2,1,0)[1,25],grad_output.permute(2,1,0)[2,36],grad_output.permute(2,1,0)[1,37],grad_output.permute(2,1,0)[1,46],grad_output.permute(2,1,0)[2,47],grad_output.permute(2,1,0)[2,69],grad_output.permute(2,1,0)[2,70],grad_output.permute(2,1,0)[2,77],grad_output.permute(2,1,0)[2,78],grad_output.permute(2,1,0)[2,87],grad_output.permute(2,1,0)[2,88],grad_output.permute(2,1,0)[2,95],grad_output.permute(2,1,0)[2,96],grad_output.permute(2,1,0)[0,97])
        #print('******************************************************************')
        return grad_output/weights, None, None #torch.where(input.abs() < 6., grad_output, 0.) 


blockgrad = blockgradfunc.apply




class prodfunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        ctx.save_for_backward(x)
        x = torch.clamp(torch.round(x), 0., 1.) 
        #x = torch.where(torch.greater_equal(x, 0.5), 1., 0.) 
        x = torch.prod(x, dim=-1)
        
        #x = torch.bernoulli(x)
        #x = torch.where((x < 0.75) * (x > 0.25), torch.clamp(torch.bernoulli(x), 0.25, 0.75), x)
        ##x = torch.where(x<1., 0.01 * x, x)
        #x = x * 0.01
        #x = (torch.sign(x-0.5)+1.)/2.
        return x


    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input = ctx.saved_tensors[0]
        
        tmp =torch.concat( [torch.ones(input.shape[0],input.shape[1],1, device = input.device),torch.cumprod(input, dim=-1)[:,:,:-1]], dim=-1) * torch.concat( [torch.flip(torch.cumprod(torch.flip(input,[2]), dim=-1),[2])[:,:,1:], torch.ones(input.shape[0],input.shape[1],1, device = input.device)], dim=-1)
        #print(grad_output, grad_output.max(), grad_output.min())
        #print(torch.abs(grad_output).max(),'here***********************')
        #print(grad_output.permute(2,1,0)[2,14],grad_output.permute(2,1,0)[1,15],grad_output.permute(2,1,0)[2,24],grad_output.permute(2,1,0)[1,25],grad_output.permute(2,1,0)[2,36],grad_output.permute(2,1,0)[1,37],grad_output.permute(2,1,0)[1,46],grad_output.permute(2,1,0)[2,47],grad_output.permute(2,1,0)[2,69],grad_output.permute(2,1,0)[2,70],grad_output.permute(2,1,0)[2,77],grad_output.permute(2,1,0)[2,78],grad_output.permute(2,1,0)[2,87],grad_output.permute(2,1,0)[2,88],grad_output.permute(2,1,0)[2,95],grad_output.permute(2,1,0)[2,96],grad_output.permute(2,1,0)[0,97])
        #print('******************************************************************')
        ####tmp = torch.clamp(tmp, 0., 1.) * 1.
        return grad_output.unsqueeze(-1) * tmp


prod = prodfunc.apply