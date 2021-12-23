import math
import random
import numpy as np
import matplotlib
from collections import namedtuple, deque

import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU, Sigmoid
from torch.nn.modules.container import Sequential
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T



class Model(nn.Module):

    def __init__(self, cols, rows):
        super(Model, self).__init__()


        layer_size = cols*rows

        self.net = Sequential(
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, cols),
            nn.Sigmoid()
        )

    # Forward pass of the model
    def forward(self, x):
        return self.net(x)






