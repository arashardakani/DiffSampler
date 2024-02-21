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
        self.Adder = Adder(8)
        # self.activation = Sgn()

    def forward(self, input):
        """x1 = self.activation.apply(self.x1)
        x2 = self.activation.apply(self.x2)
        x3 = self.activation.apply(self.x3)
        x4 = self.activation.apply(self.x4)"""
        x1, x2 = self.EMB(input)

        out, _ = self.Adder(x1, x2)

        return out


class EMB(nn.Module):
    def __init__(self):
        super().__init__()
        self.data1 = nn.Embedding(1, 8)
        self.data1.weight.data.mul_(0.01)
        self.data2 = nn.Embedding(1, 8)
        self.data2.weight.data.mul_(0.01)
        self.activation1 = Sgn()
        self.activation2 = Sgn()
        # print('Input1 before training:', self.activation1.apply(self.data1(input)))
        # print('Input2 before training:', self.activation1.apply(self.data1(input)))

    def forward(self, input):
        data1 = self.activation1.apply(self.data1(input))
        data2 = self.activation2.apply(self.data2(input))
        # print(data)
        # print(data1,data2)
        print("input1 after training:", data1.data.cpu())
        print("input2 after training:", data2.data.cpu())
        return data1, data2


device = "cuda" if torch.cuda.is_available() else "cpu"
num_train_epochs = 50
lr = 1e1
model = Circuit().to(device)


# target = torch.ones(1,8, requires_grad = False, device = device) * -1
target = torch.empty(1, 8).random_(2).to(device) * 2.0 - 1.0
print("target:", target)
loss = MSELoss()
optim = torch.optim.SGD(model.parameters(), lr=lr)  # torch.optim.

for epoch in range(num_train_epochs):
    model.train()

    optim.zero_grad()
    input = torch.LongTensor([0]).to(device)
    outputs = model(input)

    l = loss(outputs, target)
    l.backward()
    optim.step()
    print("output:", outputs.data.cpu())

print("output:", outputs)
