import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

class Policy(nn.Module):
    def __init__(self, state_shape, num_actions, hidden_layers):
        super().__init__()
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.hidden_layers = hidden_layers
        self.fcLayers = []
        for i in range(len(hidden_layers)):
            if i == 0:
                self.fcLayers.append(nn.Linear(state_shape[0], hidden_layers[i]))
            else:
                self.fcLayers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
        self.fcLayers.append(nn.Linear(hidden_layers[-1], num_actions))
        self.fcLayers = nn.ModuleList(self.fcLayers)

    def forward(self, x):
        for i in range(len(self.fcLayers) - 1):
            x = torch.relu(self.fcLayers[i](x))
        x = self.fcLayers[-1](x)
        return F.softmax(x, dim=-1)


class PGAgent:
    def __init__(self, num_actions, state_shape, hidden_layers, device):
        self.num_actions = num_actions
        self.state_shape = state_shape
        self.hidden_layers = hidden_layers
        self.use_raw = False
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.policy = Policy(
            state_shape=state_shape,
            num_actions=num_actions,
            hidden_layers=hidden_layers,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.01)

    def step(self, state):
        state = state['obs']
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action_probs = self.policy(state)
        m = torch.distributions.Categorical(action_probs)
        action = m.sample()
        return action.item()

    def eval_step(self, state):
        state = state['obs']
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action_probs = self.policy(state)
        return torch.argmax(action_probs).item(), {}

    def learn(self, states, actions, rewards, gamma=0.9):
        states = np.array(states)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        running_add = 0
        for i in reversed(range(len(rewards))):
            running_add = running_add * gamma + rewards[i]
            rewards[i] = running_add
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        action_probs = self.policy(states)
        m = torch.distributions.Categorical(action_probs)
        log_probs = m.log_prob(actions)
        loss = torch.tensor(0.).to(self.device)
        for log_prob, reward in zip(log_probs, rewards):
            loss += -log_prob * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
