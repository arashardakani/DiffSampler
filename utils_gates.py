import torch
import torch.nn as nn
import pdb
#import matplotlib.pyplot as plt
#import seaborn as sns
import math
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math

class AND(nn.Module):
    def __init__(self, num_in):
        super().__init__()
        self.dense = nn.Linear(num_in, 1)
        self.dense.weight.data.fill_(1.)
        self.dense.bias.data.fill_(-1. * num_in + 1.)
        self.dense.weight.requires_grad = False
        self.dense.bias.requires_grad = False
        self.Amp = Amp()
        self.activation = Sgn()

    def forward(self, input):
        input = self.Amp.apply(input)
        output = self.dense(input)
        output = self.activation.apply(output)
        return output


class OR(nn.Module):
    def __init__(self, num_in):
        super().__init__()
        self.dense = nn.Linear(num_in, 1)
        self.dense.weight.data.fill_(1.)
        self.dense.bias.data.fill_(1. * num_in - 1.)
        self.dense.weight.requires_grad = False
        self.dense.bias.requires_grad = False
        self.Amp = Amp()
        self.activation = Sgn()

    def forward(self, input):
        input = self.Amp.apply(input)
        output = self.dense(input)
        output = self.activation.apply(output)
        return output



class NOT(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(1, 1)
        self.dense.weight.data.fill_(-1.)
        self.dense.bias.data.fill_(0.)
        self.dense.weight.requires_grad = False
        self.dense.bias.requires_grad = False
        self.Amp = Amp()
        self.activation = Sgn()

    def forward(self, input):
        output = self.dense(input)
        output = self.activation.apply(output)
        return output




class NOR(nn.Module):
    def __init__(self, num_in):
        super().__init__()
        self.dense = nn.Linear(num_in, 1)
        self.dense.weight.data.fill_(-1.)
        self.dense.bias.data.fill_(-1. * num_in + 1.)
        self.dense.weight.requires_grad = False
        self.dense.bias.requires_grad = False
        self.Amp = Amp()
        self.activation = Sgn()

    def forward(self, input):
        input = self.Amp.apply(input)
        output = self.dense(input)
        output = self.activation.apply(output)
        return output



class NAND(nn.Module):
    def __init__(self, num_in):
        super().__init__()
        self.dense = nn.Linear(num_in, 1)
        self.dense.weight.data.fill_(-1.)
        self.dense.bias.data.fill_(1. * num_in - 1.)
        self.dense.weight.requires_grad = False
        self.dense.bias.requires_grad = False
        self.Amp = Amp()
        self.activation = Sgn()

    def forward(self, input):
        input = self.Amp.apply(input)
        output = self.dense(input)
        output = self.activation.apply(output)
        return output



class XNOR(nn.Module):
    def __init__(self, num_in):
        super().__init__()
        self.AND = AND(num_in)
        self.NOR = NOR(num_in)
        self.OR = OR(2)

    def forward(self, input):
        output1 = self.AND(input)
        output2 = self.NOR(input)
        output = self.OR(torch.concat([output1,output2], dim=-1))
        return output


class XOR(nn.Module):
    def __init__(self, num_in):
        super().__init__()
        self.OR = OR(num_in)
        self.NAND = NAND(num_in)
        self.AND = AND(2)

    def forward(self, input):
        output1 = self.OR(input)
        output2 = self.NAND(input)
        output = self.AND(torch.concat([output1,output2], dim=-1))
        return output

class bin2dec(nn.Module):
    def __init__(self, num_in):
        super().__init__()
        self.dense = nn.Linear(num_in, 1)
        self.dense.weight.data = torch.pow(2.,torch.arange(0,num_in))
        self.dense.bias.data.fill_(0.)
        self.dense.weight.requires_grad = False
        self.dense.bias.requires_grad = False

    def forward(self, input):
        output = self.dense((input+1.)/2.)
        return output


class dec2bin(nn.Module):
    def __init__(self, num_in):
        super().__init__()
        self.dec2binconversion = dec2binconversion(num_in)

    def forward(self, input):
        output = self.dec2binconversion.apply(input)
        return output


class FAdder(nn.Module):
    def __init__(self):
        super().__init__()
        self.XOR1 = XOR(2)
        self.XOR2 = XOR(2)
        self.AND1 = AND(2)
        self.AND2 = AND(2)
        self.OR = OR(2)

    def forward(self, input, C_in):
        output1 = self.XOR1(input)
        output = self.XOR2(torch.concat([output1,C_in], dim=-1))
        output3 = self.AND1(input)
        output4 = self.AND2(torch.concat([output1,C_in], dim=-1))

        C_out = self.OR(torch.concat([output3,output4], dim=-1))
        return output, C_out


class Adder(nn.Module):
    def __init__(self, num_in):
        super().__init__()
        self.items = []
        self.num_in = num_in
        self.layer = nn.ModuleList([FAdder()for _ in range(num_in)])

    def forward(self, input1, input2):
        S = torch.ones_like(input1).to(input1.device) * -1.
        for i, layer_module in enumerate(self.layer):
            if i == 0:
                S[:,0], C = layer_module(torch.concat([input1[:,0].unsqueeze(1), input2[:,0].unsqueeze(1)], dim=-1), -1. * torch.ones(1,1).to(input1.device))
            else:
                S[:,i], C = layer_module(torch.concat([input1[:,i].unsqueeze(1), input2[:,i].unsqueeze(1)], dim=-1), C)
        return S, C


class PAND(nn.Module):
    def __init__(self, num_in):
        super().__init__()
        self.items = []
        self.num_in = num_in
        self.layer = nn.ModuleList([AND(2)for _ in range(num_in)])

    def forward(self, input1, input2):
        S = torch.zeros_like(input1).to(input1.device)
        for i, layer_module in enumerate(self.layer):
            S[:,i] = layer_module(torch.concat([input1[:,i].unsqueeze(1), input2[:,i].unsqueeze(1)], dim=-1))
        return S

class POR(nn.Module):
    def __init__(self, num_in):
        super().__init__()
        self.items = []
        self.num_in = num_in
        self.layer = nn.ModuleList([OR(2)for _ in range(num_in)])

    def forward(self, input1, input2):
        S = torch.zeros_like(input1).to(input1.device)
        for i, layer_module in enumerate(self.layer):
            S[:,i] = layer_module(torch.concat([input1[:,i].unsqueeze(1), input2[:,i].unsqueeze(1)], dim=-1))
        return S


class Multiplier(nn.Module):
    def __init__(self, num_in):
        super().__init__()
        self.items = []
        self.num_in = num_in
        self.adders = nn.ModuleList([Adder(num_in)for _ in range(num_in-1)])
        self.ands = nn.ModuleList([PAND(num_in)for _ in range(num_in)])


    def forward(self, input1, input2):
        S = torch.zeros(input1.size()[0], self.num_in * 2).to(input1.device)
        for i, layer_module in enumerate(self.ands):
            if i == 0:
                tmp = layer_module( input1, input2[:,i].repeat(1,self.num_in))
                S[:,i] = tmp[:,0]
                partial1 = torch.concat([tmp[:,1:], torch.ones(1,1).to(input1.device)* -1.], dim = -1)
            else:
                partial2 = layer_module( input1, input2[:,i].repeat(1,self.num_in))

                tmp, C = self.adders[i-1](partial1, partial2)

                S[:,i] = tmp[:,0]

                partial1 = torch.concat([tmp[:,1:], C], dim = -1)
        S[:,i+1:] = partial1
        return S


class MUX2(nn.Module):
    def __init__(self, num_in):
        super().__init__()
        self.items = []
        self.num_in = num_in
        self.AND1 = PAND(num_in)
        self.AND2 = PAND(num_in)
        self.OR = POR(num_in)
        self.NOT = NOT()

    def forward(self, input1, input2, sel):
        out1 = self.AND1(input1, self.NOT(sel).repeat(1,self.num_in))
        out2 = self.AND2(input2, sel.repeat(1,self.num_in))
        out = self.OR(out1, out2)        
        return out


class MUX(nn.Module):
    def __init__(self, num_in, num_port):
        super().__init__()
        self.items = []
        self.num_in = num_in
        self.layer = {}
        self.num_port = num_port
        ports = num_port
        for i in range(int(math.log2(num_port))):
            ports = int(ports/2)
            self.layer[i] = nn.ModuleList([MUX2(num_in)for _ in range(ports)])
            

    def forward(self, input, sel):
        out = {}
        for i in range(int(math.log2(self.num_port))):
            for t, layer_module in enumerate(self.layer[i]):
                if i == 0:
                    out[t] = layer_module(input[2*t], input[2*t+1], sel[:,0].unsqueeze(1))
                else:
                    out[t] = layer_module(out[2*t], out[2*t+1], sel[:,i].unsqueeze(1))
        return out[0]


class Sgn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input #* (1 - torch.nn.functional.tanh(input[0]+0.1*torch.randn(1).to(input[0].device))**2)





class Amp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input * torch.sigmoid(grad_input.sign() * input[0] + 2.)


class dec2binconversion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, num_in):
        mask = 2 ** torch.arange(num_in - 1, -1, -1).to(input.device, input.dtype)
        ctx.save_for_backward(num_in)
        return input.unsqueeze(-1).bitwise_and(mask).ne(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        num_in = ctx.saved_tensors[0]
        
        return (grad_output * torch.pow(2.,torch.arange(0,num_in))).sum()








