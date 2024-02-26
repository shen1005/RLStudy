import gym
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import collections

class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high, hidden_dim=128):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.var = 3
        self.action_low = action_low
        self.action_high = action_high

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return torch.tanh(x) * 2

    # 添加噪声
    def choose_action(self, state):
        self.var *= 0.995
        action = self.forward(state)[0].detach().numpy()
        return np.clip(np.random.normal(action, self.var), self.action_low, self.action_high)


class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], 1)))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


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


class DDPG:
    def __init__(self, state_dim, action_dim, action_low, action_high, gamma=0.9, lr=0.01):
        self.actorNet = ActorNet(state_dim, action_dim, action_low, action_high)
        self.criticNet = CriticNet(state_dim, action_dim)
        self.targetActorNet = ActorNet(state_dim, action_dim, action_low, action_high)
        self.targetCriticNet = CriticNet(state_dim, action_dim)
        self.optim_Actor = torch.optim.Adam(self.actorNet.parameters(), lr=lr)
        self.optim_Critic = torch.optim.Adam(self.criticNet.parameters(), lr=lr)
        self.gamma = gamma
        self.buffer = Buffer()

        self.targetCriticNet.load_state_dict(self.criticNet.state_dict())
        self.targetActorNet.load_state_dict(self.actorNet.state_dict())

    def push(self, states, actions, rewards, next_states, done):
        self.buffer.push(states, actions, rewards, next_states, done)

    def update(self, batch_size=128):
        states, actions, rewards, next_states, dones = self.buffer.getDatas(batch_size)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        acs = self.actorNet(states)
        qs = self.criticNet(states, acs)
        actor_loss = -torch.mean(qs)

        self.optim_Actor.zero_grad()
        actor_loss.backward()
        self.optim_Actor.step()

        now_critic = self.criticNet(states, actions)
        next_action = self.targetActorNet(next_states)
        critic_target = rewards + self.gamma * self.targetCriticNet(next_states, next_action) * (1 - dones)
        critic_loss = F.mse_loss(now_critic, critic_target)

        self.optim_Critic.zero_grad()
        critic_loss.backward()
        self.optim_Critic.step()
        print(actor_loss + critic_loss)

    def reset(self):
        tau = 0.005  # 软更新参数，可以调整
        for target_param, local_param in zip(self.targetCriticNet.parameters(), self.criticNet.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        return self.actorNet.choose_action(state)


env = gym.make("Pendulum-v1", render_mode="human")
print(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.low, env.action_space.high)
ddpg = DDPG(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.low, env.action_space.high)
for i in range(100):
    observation = env.reset()[0]
    times = 0
    while True:
        times += 1
        action = ddpg.choose_action(observation)
        next_observation, reward, done, info, _ = env.step(action)
        ddpg.push(observation, action, reward, next_observation, done)
        observation = next_observation
        if times % 10 == 0:
            ddpg.update()
            ddpg.reset()
        if done:
            break
env.close()





