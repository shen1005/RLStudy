import gym
import numpy as np
import time
# 查看价值
class Agent:
    def __init__(self, env):
        self.actions = list(range(env.action_space.n))
        self.policy = np.zeros(env.observation_space.n)
        self.env = env

    def choose_action(self, observation):
        position=observation
        action = self.policy[position]
        return int(action)


    def learnByValue(self, theta=0.001, discount_factor=0.9, max_iteration=1000):
        Q_before = np.zeros((env.observation_space.n, env.action_space.n))

        for i in range(max_iteration):
            Q_tmp = np.zeros((self.env.observation_space.n, self.env.action_space.n))
            delta = 0
            for state in range(self.env.observation_space.n):
                for action in range(self.env.action_space.n):
                    now_reward = 0
                    dis_reward = 0
                    for prob, next_state, reward, done in self.env.env.P[state][action]:
                        now_reward += prob * reward
                        dis_reward += prob * np.max(Q_before[next_state])
                    Q_tmp[state][action] = now_reward + discount_factor * dis_reward
                    delta = max(delta, abs(Q_tmp[state][action] - Q_before[state][action]))
            Q_before = Q_tmp
            if delta < theta:
                break
        # 根据Q值更新policy
        for state in range(self.env.observation_space.n):
            self.policy[state] = int(np.argmax(Q_before[state]))
        print(Q_before)

    def learnByPolicy(self, theta=0.001, discount_factor=0.9, max_iteration=1000):
        # 随机初始化policy
        self.policy = np.random.randint(0, 4, size=self.env.observation_space.n)
        for i in range(max_iteration):
            V = np.zeros(self.env.observation_space.n)
            delta = 0
            # 首先进行策略评估
            for j in range(max_iteration):
                V_tmp = np.zeros(self.env.observation_space.n)
                for state in range(self.env.observation_space.n):
                    v = 0
                    for prob, next_state, reward, done in self.env.env.P[state][self.policy[state]]:
                        v += prob * (reward + discount_factor * V[next_state])
                    delta = max(delta, abs(v - V[state]))
                    V_tmp[state] = v
                V = V_tmp
                if delta < theta:
                   break
            # 策略改进
            Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
            for state in range(self.env.observation_space.n):
                for action in range(self.env.action_space.n):
                    for prob, next_state, reward, done in self.env.env.P[state][action]:
                        Q[state][action] += prob * (reward + discount_factor * V[next_state])
            # 根据Q值更新policy
            flag = False
            for state in range(self.env.observation_space.n):
                action = np.argmax(Q[state])
                if action != self.policy[state]:
                    flag = True
                    self.policy[state] = action
            if not flag:
                print("success learn")
                break

env = gym.make('FrozenLake-v1', render_mode='human')
observation = env.reset()[0]
print(env.action_space)
print(env.observation_space)
print(observation)
agent = Agent(env)
agent.learnByValue()
print(agent.policy)
agent.learnByPolicy()
print(agent.policy)
# 查看策略





