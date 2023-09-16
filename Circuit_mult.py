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
        self.Mult = Multiplier(4)
        #self.activation = Sgn()

    def forward(self, input):
        '''x1 = self.activation.apply(self.x1)
        x2 = self.activation.apply(self.x2)
        x3 = self.activation.apply(self.x3)
        x4 = self.activation.apply(self.x4)'''
        
        x1, x2 = self.EMB(input)

        out = self.Mult(x1,x2)
        '''print('x1:', x1)
        print('x2:', x2)
        print('out:', out)'''
        
        return out


class EMB(nn.Module):
    def __init__(self):
        super().__init__()
        self.data1 = nn.Embedding(1, 4)
        #self.data1.weight.data.mul_(0.01)
        #self.data1.requires_grad = False
        self.data2 = nn.Embedding(1, 4)
        #self.data2.weight.data.mul_(0.01)
        self.activation1 = Sgn()
        self.activation2 = Sgn()

    def forward(self, input):
        data1 = self.activation1.apply(self.data1(input))
        data2 = self.activation2.apply(self.data2(input))
        #print(data)
        #print(data1,data2)

        return data1, data2



device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_train_epochs = 1000
lr = 1e1
model = Circuit().to(device)



#target = torch.ones(1,8, requires_grad = False, device = device) * -1
target = torch.empty(1,8).random_(2).to(device) * 2. - 1.
print('target:', target)
loss = MSELoss(reduction='sum')
optim = torch.optim.SGD(model.parameters(), lr=lr) #torch.optim.

for epoch in range(num_train_epochs):
    

    model.train()

    optim.zero_grad()
    input = torch.LongTensor([0]).to(device)
    outputs = model(input)
    
    l = loss(outputs, target)
    l.backward()
    optim.step()
    print('output:', outputs)

print('target:', target)


