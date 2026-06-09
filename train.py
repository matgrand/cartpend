import torch, torch.nn as nn
import numpy as np
from tqdm import tqdm
import cart

HID_SIZE = 12
EMB_SIZE = 16

N_STEPS = 25

BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 100


class Swish(Module): # swish
    def __init__(self, no=1): 
        super(Swish, self).__init__()
        if no == 1: self.beta = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True, dtype=torch.float32) # scalar
        else: self.beta = torch.nn.Parameter(torch.ones(no), requires_grad=True, dtype=torch.float32) # n is the number of outputs
    def forward(self, x): return x*torch.sigmoid(self.beta*x)


class CartNN(nn.Module):
    def __init__(self, n_arms=cart.N_ARMS, n_steps=N_STEPS):
        super().__init__()
        self.mlpx = nn.Sequential(
            nn.Linear(2*(n_arms+1), HID_SIZE), Swish(),
            nn.Linear(HID_SIZE, EMB_SIZE), Swish(),
        )
        self.mlpu = nn.Sequential(
            nn.Linear(n_steps, HID_SIZE), Swish(),
            nn.Linear(HID_SIZE, EMB_SIZE), Swish(),
        ) 
        self.mlph = nn.Sequential(
            nn.Linear(EMB_SIZE, HID_SIZE), Swish(),
            nn.Linear(HID_SIZE, 2*(n_arms+1)), Swish(),
        )
    def forward(self, x, u_seq):
        x_emb = self.mlpx(x)
        u_emb = self.mlpu(u_seq)
        h_emb = x_emb * u_emb
        return self.mlph(h_emb)


def dataset(n_samples=1000, n_arms=cart.N_ARMS, n_steps=N_STEPS):
    x = torch.randn(n_samples, 2*(n_arms+1))
    u_seq = torch.randn(n_samples, n_steps)
    y = torch.randn(n_samples, 2*(n_arms+1))
    return x, u_seq, y


if __name__ == "__main__":
    pass