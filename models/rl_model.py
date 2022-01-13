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
        x = x.flatten(-2)

        return self.net(x)

# class Model(nn.Module):
#
#     def __init__(self, cols, rows):
#         super(Model, self).__init__()
#         n_conv_layers = 2
#         conv_channels = [(3**i, 3**(i+1)) for i in range(n_conv_layers)]
#         def conv2d_size_out(size, kernel_size=5, stride=1, padding=2):
#             return (size - (kernel_size - 1) - 1) // stride + 1 + 2 * padding
#         convw = cols
#         convh = rows
#         for i in range(n_conv_layers):
#             convw = conv2d_size_out(convw)
#             convh = conv2d_size_out(convh)
#         linear_input_size = convw * convh * (conv_channels[-1][1] if n_conv_layers > 0 else 1)
#         self.convs = nn.ModuleList([nn.Conv2d(conv_channels[i][0], conv_channels[i][1], kernel_size=5, stride=1, padding=2) for i in range(n_conv_layers)])
#         self.bns = nn.ModuleList([nn.BatchNorm2d(conv_channels[i][1]) for i in range(n_conv_layers)])
#         # conv_bn_relu_stack = [item for tup in zip(convs, bns, relus) for item in tup]
#         # self.net = Sequential(
#         #     *conv_bn_relu_stack,
#         # )
#         # print(self.net)
#         self.hidden_linear = nn.Linear(linear_input_size, linear_input_size)
#         self.head = nn.Linear(linear_input_size, cols)
#
#     # Forward pass of the model
#     def forward(self, x):
#         x = torch.unsqueeze(x, 1)
#         for i in range(len(self.convs)):
#             x = F.relu(self.bns[i](self.convs[i](x)))
#         x = F.relu(self.hidden_linear(x.view(x.size(0), -1)))
#         x = self.head(x)
#         # x = torch.sigmoid(x)
#         return x







