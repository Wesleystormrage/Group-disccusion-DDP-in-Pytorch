import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def demo_basic():
    pid=os.getpid()
    print(f"Running basic DDP example on pid {pid}.")
    model = ToyModel().to("cuda:0")

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs =model(torch.randn(20, 10).to("cuda:0"))
    labels = torch.randn(20, 5).to("cuda:0")
    loss_fn(outputs, labels).backward()
    optimizer.step()
    print("Single device training finished")

if __name__=="__main__":
    demo_basic()