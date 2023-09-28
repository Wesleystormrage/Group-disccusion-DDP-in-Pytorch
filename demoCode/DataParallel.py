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
    print(f"Running basic DDP example on pid {os.getpid()}.")
    model = ToyModel()
    device_ids=[0,1]
    model_parallel=torch.nn.DataParallel(model,device_ids=device_ids)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model_parallel.parameters(), lr=0.001)
    optimizer.zero_grad()
    outputs =model(torch.randn(20, 10))
    labels = torch.randn(20, 5)
    loss_fn(outputs, labels).backward()
    optimizer.step()

if __name__=="__main__":
    demo_basic()