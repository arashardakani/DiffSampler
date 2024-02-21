import torch
import torch.nn as nn
import math
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
from utils_pgates import *
import argparse
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
from pysat.formula import CNF


class Circuit(nn.Module):
    def __init__(self, problems):
        super().__init__()
        self.EMB = EMB(problems.nv)
        self.layer = {}
        self.layer = nn.ModuleList([OR() for i in range(len(problems.clauses))])
        self.problems = problems

    def forward(self, input):
        intermediate_out = torch.zeros(input.size()[0], len(self.problems.clauses)).to(
            input.device
        )
        x = self.EMB(input)
        for i in range(len(self.problems.clauses)):
            idx = [abs(x) - 1 for x in self.problems.clauses[i]]
            y = torch.where(
                torch.FloatTensor(self.problems.clauses[i]).to(x.device) > 0.0,
                x[:, idx],
                1.0 - x[:, idx],
            )
            intermediate_out[:, i] = self.layer[i](y)

        return intermediate_out


class EMB(nn.Module):
    def __init__(self, num_in):
        super().__init__()
        self.data = nn.Embedding(1, num_in)
        # self.data.weight.data.mul_(0.01)
        self.activation = torch.nn.Sigmoid()

    def forward(self, input):
        data = self.data(input)
        # print(data)
        out = self.activation(data)
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
        default=10e-1,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,
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
    print("\n\nCNF File Name:", args.cnf_name_or_path)
    print("No. clauses:", len(problem.clauses))
    print("No. variables", problem.nv)
    print("Learning Rate:", args.learning_rate)
    print("No. Epochs:", args.num_train_epochs)
    print("\n*****************Training*****************\n\n")
    model = Circuit(problem).to(device)

    target = torch.ones(1, len(problem.clauses), requires_grad=False, device=device)

    loss = MSELoss(reduction="sum")
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  # torch.optim.

    for epoch in range(args.num_train_epochs):
        model.train()

        optim.zero_grad()
        input = torch.LongTensor([0]).to(device)
        outputs = model(input)
        l = loss(outputs, target)
        print("Loss value @ epoch", epoch, ":", (torch.round(l * 32.0) / 32.0).item())
        l.backward()
        optim.step()
    print(
        "\nSolution:",
        ((torch.sign(model.EMB.data.weight.data) + 1.0) / 2.0).long().cpu().tolist(),
        "\n",
    )


if __name__ == "__main__":
    main()
