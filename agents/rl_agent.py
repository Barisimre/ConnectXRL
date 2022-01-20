from agents.agent import Agent
from models.rl_customloss import CustomLoss
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

    def __init__(self, configuration) -> None:
        super().__init__(configuration)
        self.training = True
        self.buffer = ReplayMemory(500)
        self.training_threshold = 400
        self.batch_size = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.learnt = False
        self.moves_made = 0
        self.exploration_start = 0.9
        self.exploration_end = 0.05
        self.exploration_decay = 400

        self.policy_net = Model(self.configuration.columns, self.configuration.rows).to(self.device)
        self.policy_net.train()
        self.target_net = Model(self.configuration.columns, self.configuration.rows).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.gamma = 0.95
        # TODO check this, the tutorial uses a different one
        self.optimizer = optim.RMSprop(self.policy_net.parameters())

    def make_move(self, observation, configuration):
        state = torch.tensor([self.state_in_2d(observation.board)], device=self.device, dtype=torch.float)
        value = None
        if self.training:
            sample = random.random()
            exploration_likelihood = self.exploration_end + (self.exploration_start - self.exploration_end) * math.exp(
                -1. * self.moves_made / self.exploration_decay)
            if sample < exploration_likelihood:
                action = random.randrange(self.configuration.columns)
            else:
                with torch.no_grad():
                    net_output = self.policy_net(state).max(1)
                    value = net_output[0].item()
                    action = net_output[1].item()
        else:
            with torch.no_grad():
                net_output = self.policy_net(state).max(1)
                value = net_output[0].item()
                action = net_output[1].item()
        self.moves_made += 1
        # print(f'action: {action}, value: {value}, moves: {self.moves_made}')
        return action

    def optimize(self):

        # We have enough in the buffer, update some parameters
        if len(self.buffer) < self.training_threshold:
            return
        self.learnt = True
        transitions = self.buffer.sample(self.batch_size)

        batch = Transition(*zip(*transitions))
        non_final_mask = \
            torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = \
            torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_values = self.policy_net(state_batch)
        state_action_values = state_values.gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # expected_state_action_values.clamp_(-1,1)
        criterion = nn.SmoothL1Loss()

        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # print(f"loss:         {loss}")
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)  # not clear why we need this. Is this just a common thing?
        self.optimizer.step()
        # with torch.no_grad():
        #     updated_state_values = self.policy_net(state_batch)
        #     updated_state_action_values = updated_state_values.gather(1, action_batch)
        # updated_loss = criterion(updated_state_action_values, expected_state_action_values.unsqueeze(1))
        # print(f'updated_loss: {updated_loss}')
        # print(f'updated_state_action_values: {updated_state_action_values.squeeze()}')
        # self.buffer.empty()

    def update_networks(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # to ensure the target net stays in eval mode, because load_state_dict might affect that

    # Add to the replay buffer
    def save(self, state, action, reward, new_state):
        # print(state_in_2d(old_state))
        self.buffer.push(torch.tensor([self.state_in_2d(state)], device=self.device, dtype=torch.float),
                         torch.tensor([[action]], device=self.device, dtype=torch.long),
                         torch.tensor([reward], device=self.device, dtype=torch.float),
                         torch.tensor([self.state_in_2d(new_state)], device=self.device, dtype=torch.float)
                         if new_state is not None else None)

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def state_in_2d(self, state):
        columns = self.configuration.columns
        rows = self.configuration.rows
        state_2d = [state[r * columns: (r + 1) * columns] for r in range(rows)]
        return state_2d


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def empty(self):
        self.memory = deque([], maxlen=self.capacity)
