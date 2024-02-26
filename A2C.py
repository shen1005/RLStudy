import gym
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
from torch import nn

class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

class CriticNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(CriticNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class A2C:
    def __init__(self, state_dim, action_dim, hidden_dim=128, gamma=0.9, lr=0.01):
        self.actor = ActorNet(state_dim, action_dim, hidden_dim)
        self.critic = CriticNet(state_dim, hidden_dim)
        self.gamma = gamma
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        #print(state.shape)
        probs = self.actor(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action).squeeze(0)

    def update(self, state, log_prob, next_state, reward):
        state = torch.tensor(state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        # Q网络
        value = self.critic(state)
        next_value = self.critic(next_state)
        target_value = reward + self.gamma * next_value

        self.optimizer_critic.zero_grad()
        loss1 = F.mse_loss(value, target_value)
        #self.optimizer_critic.step()

        ratio = reward + self.gamma * next_value - value
        self.optimizer_actor.zero_grad()
        loss2 = -ratio * log_prob + loss1
        loss2.backward()
        self.optimizer_actor.step()
        self.optimizer_critic.step()
        print(loss1, loss2)
        return loss1, loss2


env = gym.make('CartPole-v1', render_mode="human")
a2c = A2C(env.observation_space.shape[0], env.action_space.n)
for i in range(200):
    observation = env.reset()[0]
    env.render()
    while True:
        action, log_prob = a2c.choose_action(observation)
        next_observation, reward,  done, info, _ = env.step(action)
        a2c.update(observation, log_prob, next_observation, reward)
        if done:
            break
env.close()