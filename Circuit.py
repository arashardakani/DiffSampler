import torch
import torch.nn as nn
import math
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
from utils_gates import *

from torch.nn import CrossEntropyLoss, MSELoss


class Circuit(nn.Module):
    def __init__(self):
        super().__init__()
        self.EMB = EMB()
        self.x1 = nn.parameter.Parameter(torch.ones(1, 1) * 0.1)
        self.x2 = nn.parameter.Parameter(torch.ones(1, 1) * 0.1)
        self.x3 = nn.parameter.Parameter(torch.ones(1, 1) * 0.1)
        self.x4 = nn.parameter.Parameter(torch.ones(1, 1) * 0.1)
        self.NAND1 = NAND(2)
        self.NAND2 = NAND(2)
        self.NAND3 = NAND(2)
        self.NAND4 = NAND(2)
        self.NAND5 = NAND(2)
        self.NAND6 = NAND(2)
        # self.activation = Sgn()

    def forward(self, input):
        """x1 = self.activation.apply(self.x1)
        x2 = self.activation.apply(self.x2)
        x3 = self.activation.apply(self.x3)
        x4 = self.activation.apply(self.x4)"""
        x1, x2, x3, x4 = self.EMB(input)

        out1 = self.NAND1(torch.concat([x1, x2], dim=-1))
        out2 = self.NAND2(torch.concat([x3, x4], dim=-1))

        out3 = self.NAND3(torch.concat([out1, out2], dim=-1))

        out4 = self.NAND4(torch.concat([out1, out3], dim=-1))
        out5 = self.NAND5(torch.concat([out2, out3], dim=-1))

        out6 = self.NAND6(torch.concat([out4, out5], dim=-1))

        return out6


class EMB(nn.Module):
    def __init__(self):
        super().__init__()
        self.data = nn.Embedding(1, 4)
        self.data.weight.data.mul_(0.01)
        """self.x1 = nn.Parameter(torch.ones(1,1)*0.1, requires_grad = True)
        self.x2 = nn.Parameter(torch.ones(1,1)*0.1, requires_grad = True)
        self.x3 = nn.Parameter(torch.ones(1,1)*0.1, requires_grad = True)
        self.x4 = nn.Parameter(torch.ones(1,1)*0.1, requires_grad = True)"""
        self.activation = Sgn()

    def forward(self, input):
        data = self.data(input)
        # print(data)
        x1 = self.activation.apply(data[0, 0].unsqueeze(0).unsqueeze(1))
        x2 = self.activation.apply(data[0, 1].unsqueeze(0).unsqueeze(1))
        x3 = self.activation.apply(data[0, 2].unsqueeze(0).unsqueeze(1))
        x4 = self.activation.apply(data[0, 3].unsqueeze(0).unsqueeze(1))

        return x1, x2, x3, x4


device = "cuda" if torch.cuda.is_available() else "cpu"
num_train_epochs = 10
lr = 1e-1
model = Circuit().to(device)


x1 = torch.ones(1, 1, requires_grad=True, device=device) * 0.1
x2 = torch.ones(1, 1, requires_grad=True, device=device) * 0.1
x3 = torch.ones(1, 1, requires_grad=True, device=device) * 0.1
x4 = torch.ones(1, 1, requires_grad=True, device=device) * 0.1

target = torch.ones(1, 1, requires_grad=False, device=device) * -1

loss = MSELoss()
optim = torch.optim.SGD(model.parameters(), lr=lr)  # torch.optim.

for epoch in range(num_train_epochs):
    model.train()

    optim.zero_grad()
    input = torch.LongTensor([0]).to(device)
    outputs = model(input)
    l = loss(outputs, target)
    print(l)
    l.backward()
    optim.step()
