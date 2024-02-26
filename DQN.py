import random

import torch
from torch.nn import functional as F
from torch import nn
import collections
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Buffer:
    def __init__(self, max_size=128):
        self.buffer = collections.deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        # 队列满了就把最早的数据删掉
        self.buffer.append((state, action, reward, next_state, done))

    def getDatas(self, batch_size, isRandom=True):
        # 随机采样
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        batch_size = int(batch_size)
        if isRandom:
            datas = random.sample(self.buffer, batch_size)
        else:
            # 取最近的数据
            datas = [self.buffer[-i] for i in range(1, batch_size + 1)]

        states, actions, rewards, next_states, dones = zip(*datas)
        return states, actions, rewards, next_states, dones

class DQN:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=0.01, gamma=0.9, buffer_max_size=128):
        self.Net = MLP(state_dim, action_dim, hidden_dim)
        self.targetNet = MLP(state_dim, action_dim, hidden_dim)
        self.buffer = Buffer(buffer_max_size)
        self.optimizer = torch.optim.Adam(self.Net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = np.linspace(0.9, 0.1, num=1000)
        self.sample_times = 0
        self.action_dim = action_dim
        self.count = 0

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        if np.random.random() < self.epsilon[self.sample_times]:
            action = np.random.randint(0, self.action_dim)
        else:
            action = torch.argmax(self.Net(state)).item()
        self.sample_times += 1
        if self.sample_times >= 1000:
            self.sample_times = 999
        return action

    def update(self, batch_size=32):
        states, actions, rewards, next_states, dones = self.buffer.getDatas(batch_size)
        # 变成tensor
        states = torch.tensor(states, dtype=torch.float32).unsqueeze(1)
        actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # 原来的Q值
        q = self.Net(states).gather(1, actions)
        # 目标Q值, 用targetNet计算, 还要取最大值
        max_q = self.targetNet(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + self.gamma * max_q * (1 - dones)
        # 计算loss

        loss = F.mse_loss(q, target_q)
        # 更新网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.count += 1
        # 更新targetNet
        if self.count % 10 == 0:
            self.targetNet.load_state_dict(self.Net.state_dict())

buffer = Buffer(10)
for i in range(20):
    buffer.push(i%3, i%3, i%3, i%3, i%3)
states, actions, rewards, next_states, dones = buffer.getDatas(10)
net = MLP(1, 3)
target_Net = MLP(1, 3)
states = torch.tensor(states, dtype=torch.float32).unsqueeze(1)
print(torch.max(net(states), 1))
print(torch.max(net(states), 1)[0])
print(torch.max(net(states), 1)[0].unsqueeze(1))
print(torch.max(net(states), 1)[1])
print(torch.max(net(states), 1)[1].unsqueeze(1))
