from agents.agent import Agent
from models.rl_model import Model
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

class RLAgent(Agent):

    def __init__(self, cols, rows) -> None:
        super().__init__()

        self.training = True
        self.buffer = ReplayMemory(1000)
        self.training_threshold = 100
        self.batch_size = 100
        self.loss_function = nn.SmoothL1Loss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.learnt = False

        self.policy_net = Model(cols, rows).to(self.device)
        self.target_net = Model(cols, rows).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.gamma = 0.999
        # TODO check this, the tutorial uses a different one
        self.optimizer = optim.Adam(self.policy_net.parameters())


    def make_move(self, observation):
        if not self.learnt:
            return random.randint(0, 6)
        state = torch.tensor([observation.board], device=self.device, dtype=torch.float)
        action = self.policy_net(state).argmax(1)[0].detach().item()
        # print(action)
        return action


    def optimize(self):
        # We have enough in the buffer, update some parameters
        if len(self.buffer) < self.training_threshold:
            return
        print("Optimize")
        self.learnt = True
        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        next_state_batch = torch.cat([s for s in batch.next_state
                                           if s is not None])

        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.loss_function(state_action_values, expected_state_action_values.unsqueeze(1))
        print(f"Loss: {loss}")

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1,1) # no clue what this does
        self.optimizer.step()
        self.buffer.empty()

    def update_networks(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


    # Add to the replay buffer
    def save(self, old_obs, action, reward, new_obs):
        self.buffer.push(torch.tensor([old_obs], device=self.device, dtype=torch.float),
                         torch.tensor([[action]], device=self.device),
                         torch.tensor([reward], device=self.device),
                         torch.tensor([new_obs], device=self.device, dtype=torch.float))

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def empty(self):
        self.memory = deque([],maxlen=self.capacity)