import torch
import torch.nn as nn
import math
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
from utils_gates import *
import argparse
from torch.nn import CrossEntropyLoss, MSELoss
from pysat.formula import CNF


class Circuit(nn.Module):
    def __init__(self, problems):
        super().__init__()
        self.EMB = EMB(problems.nv)
        self.layer = {}
        self.layer = nn.ModuleList(
            [OR(len(problems.clauses[i])) for i in range(len(problems.clauses))]
        )
        for i in range(len(problems.clauses)):
            for j in range(len(problems.clauses[i])):
                self.layer[i].dense.weight.data[0, j].mul_(
                    torch.sign(torch.tensor(problems.clauses[i][j]))
                )
        self.AND = AND(len(problems.clauses))
        self.problems = problems

    def forward(self, input):
        intermediate_out = torch.zeros(input.size()[0], len(self.problems.clauses)).to(
            input.device
        )
        x = self.EMB(input)
        for i in range(len(self.problems.clauses)):
            idx = [abs(x) - 1 for x in self.problems.clauses[i]]
            """print('I am here')
            print(x.shape)
            print(idx, x[:,idx])"""
            intermediate_out[:, i] = self.layer[i](x[:, idx])
        print(intermediate_out)
        out = self.AND(intermediate_out)
        return out


class EMB(nn.Module):
    def __init__(self, num_in):
        super().__init__()
        self.data = nn.Embedding(1, num_in)
        # self.data.weight.data.mul_(0.01)
        self.data.weight.data[0, 0] = 1.0
        self.data.weight.data[0, 1] = -1.0
        self.data.weight.data[0, 2] = -1.0
        self.data.weight.data[0, 3] = 1.0

        self.activation = Sgn()

    def forward(self, input):
        data = self.data(input)
        print(data)
        out = self.activation.apply(data)
        return out


device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--cnf_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-2,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=10,
        help="Total number of training epochs to perform.",
    )
    """parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)"""

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    problem = CNF(from_file=args.cnf_name_or_path)

    model = Circuit(problem).to(device)

    target = torch.ones(1, 1, requires_grad=False, device=device)

    loss = MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  # torch.optim.

    for epoch in range(args.num_train_epochs):
        model.train()

        optim.zero_grad()
        input = torch.LongTensor([0]).to(device)
        outputs = model(input)
        l = loss(outputs, target)
        print(l)
        l.backward()
        optim.step()


if __name__ == "__main__":
    main()
