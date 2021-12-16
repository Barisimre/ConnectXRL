import math
import random
import numpy as np
import matplotlib


import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU, Sigmoid
from torch.nn.modules.container import Sequential
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):

    def __init__(self, cols, rows):
        super(Model, self).__init__()

        self.buffer = []
        self.treshold = 100

        layer_size = cols*rows+1

        self.net = Sequential(
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        return self.net(x)


    def optimize(self):
        pass

    def save(self, old_obs, action, reward, new_obs):
        self.buffer.append([old_obs, action, reward, new_obs])
        if len(self.buffer) > self.treshold:
            self.optimize()
            self.buffer = []
    