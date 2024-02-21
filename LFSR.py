import torch
import torch.nn as nn
import math
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
from utils_gates import *

from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim.lr_scheduler import LinearLR


class Circuit(nn.Module):
    def __init__(self):
        super().__init__()
        self.XOR1 = XOR(2)
        self.reg = torch.zeros(1, 22)
        self.activation = Sgn()

    def forward(self, input):
        """x1 = self.activation.apply(self.x1)
        x2 = self.activation.apply(self.x2)
        x3 = self.activation.apply(self.x3)
        x4 = self.activation.apply(self.x4)"""
        self.reg = input
        # print(self.reg[0,5:7].shape)

        x1 = self.activation.apply(self.reg[0, 20:22].unsqueeze(0))
        # x2 = self.activation.apply(self.reg[0,6].unsqueeze(0).unsqueeze(1))
        out1 = self.XOR1(x1)

        # out1 = self.XOR1(torch.concat([x1,x2], dim = -1)) #* 0.1
        out = torch.concat([out1, self.reg[0, 0:21].unsqueeze(0)], dim=-1)
        return out

    def reset_reg(self):
        self.reg.zero_()


class EMB(nn.Module):
    def __init__(self):
        super().__init__()
        self.data = nn.Embedding(1, 22)
        self.data.weight.data.mul_(0.01)
        self.activation = Sgn()

    def forward(self, input):
        data = self.data(input)
        # print(data)

        return data


device = "cuda" if torch.cuda.is_available() else "cpu"
num_train_epochs = 10000
lr = 1e0
model = Circuit().to(device)
input_model = EMB().to(device)
num_clk_cycle = 21


target = torch.ones(1, 22, requires_grad=False, device=device) * -1.0
target[0, 5] = 1.0
target[0, 2] = 1.0

output_func = Sgn()

print(target)

loss = MSELoss()
optim = torch.optim.Adam(
    [{"params": model.parameters()}, {"params": input_model.parameters()}], lr=lr
)  # torch.optim.

scheduler = LinearLR(
    optim, start_factor=1.0, end_factor=0.0000005, total_iters=num_train_epochs
)
input = torch.ones(1, 7, requires_grad=True, device=device) * 0.1


for epoch in range(num_train_epochs):
    model.train()
    input_model.train()
    optim.zero_grad()
    # torch.LongTensor([0]).to(device)
    input = input_model(torch.LongTensor([0]).to(device))
    # print(input)
    # model.reset_reg()
    outputs = model(input)
    for i in range(num_clk_cycle - 1):
        outputs = model(outputs)

    print(outputs)
    l = loss(output_func.apply(outputs), target)
    # print(l)
    l.backward()
    optim.step()
    scheduler.step()
    # print('end of update')
