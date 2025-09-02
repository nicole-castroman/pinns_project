import torch.nn as nn
import torch

def get_activation(name: str):
    name = (name or "tanh").lower()
    return {"tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU()}.get(name, nn.Tanh())

class DNN(nn.Module):
    """Fully-connected MLP with Xavier init."""
    def __init__(self, layers, activation="tanh"):
        super().__init__()
        act = get_activation(activation)
        mods = []
        for i in range(len(layers)-1):
            linear = nn.Linear(layers[i], layers[i+1])
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            mods.append(linear if i==len(layers)-2 else nn.Sequential(linear, act))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)
