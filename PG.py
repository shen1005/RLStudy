import argparse
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from RlCardTest import deZhouEnv
import wandb
import matplotlib.pyplot as plt

class PGNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PGNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.save_log_probs = []
        self.rewards = []
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs = self.forward(state)
        m = Categorical(action_probs)
        action = m.sample()
        self.save_log_probs.append(m.log_prob(action).squeeze(0))
        return action.item()

    def update(self, gamma=0.9):
        running_add = 0
        for i in reversed(range(len(self.rewards))):
            running_add = running_add * gamma + self.rewards[i]
            self.rewards[i] = running_add
        # normalize rewards
        rewards = torch.tensor(self.rewards)
        # if len(rewards) > 1:
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-9)
        # 开始更新
        loss = torch.tensor(0.)
        for log_prob, reward in zip(self.save_log_probs, rewards):
            loss += -log_prob * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print("loss: ", loss)
        self.save_log_probs = []
        self.rewards = []

    def ppoUpdate(self, states, actions, old_log_probs, rewards, epsilon=0.2):
        running_add = 0
        for i in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[i]
            rewards[i] = running_add
        # normalize rewards
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-9)
        # 计算新的log_probs
        action_probs = self.forward(states)
        m = Categorical(action_probs)
        new_log_probs = m.log_prob(actions)
        # 计算ratio
        ratio = torch.exp(new_log_probs - old_log_probs) # 输入的old_log_probs是一个必须已经是tensor，如果是list，需要用torch.cat(old_log_probs)
        # 计算surrogate loss
        surr1 = ratio * rewards
        surr2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * rewards
        loss = -torch.min(surr1, surr2).mean()
        # 开始更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print("loss: ", loss)


env = deZhouEnv()
torch.manual_seed(4)
print('observation space:', env.observation_space)
print('action space:', env.action_space)
print(env.observation_space.shape[0], env.action_space.n)
agent = PGNet(env.observation_space.shape[0], env.action_space.n)
# agent = torch.load("model.pth")
agent.load_state_dict(torch.load("model.pth"))
rewards = []
ep = 0
while ep < 1000:
    observation = env.reset()
    for t in range(1000):
        #env.render()
        action = agent.choose_action(observation)
        observation, reward, done, info, _ = env.step(action)
        print(f"action: {action}")
        print(action, reward, done)
        agent.rewards.append(reward)
        if done:
            rewards.append(reward)
            print("Episode finished after {} timesteps".format(t+1))
            agent.update(gamma=0.5)
            break
    ep += 1
print(rewards)
env.close()
plt.plot(rewards)
plt.show()
# 统计前100个episode的reward大于0的episode的数量
# print(len([i for i in rewards[:10] if i > 0]))
# # 统计前100个episode的reward小于0的episode的数量
# print(len([i for i in rewards[-10:] if i > 0]))
print(np.mean(rewards[:50]))
print(np.mean(rewards[-50:]) )
inp = input("是否保存模型？")
if inp == "y":
    torch.save(agent.state_dict(), "model2.pth")
