import gym
import numpy as np


# epsilon-greedy
class Agent:
    def __init__(self, action_num, alpha=0.1, gamma=0.9):
        self.Q = {}
        self.alpha = alpha
        self.gamma = gamma
        self.action_num = action_num
        # 回合数

    def sample_action(self, state, epsilon=0.1):
        if state not in self.Q.keys():
            self.Q[state] = np.zeros(self.action_num)
            return np.random.randint(0, self.action_num)
        else:
            if np.random.random() < epsilon:
                return np.random.randint(0, self.action_num)
            else:
                return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        if done:
            self.Q[state][action] += self.alpha * (reward - self.Q[state][action])
        else:
            if next_state not in self.Q.keys():
                self.Q[next_state] = np.zeros(self.action_num)
            self.Q[state][action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])

    def get_Q(self):
        return self.Q


if __name__ == "__main__":
    env = gym.make("CliffWalking-v0", render_mode="human")
    max_iteration = 600
    agent = Agent(env.action_space.n)
    rewards = []
    for i_ep in range(max_iteration):
        reward_sum = 0
        observation = env.reset()[0]
        epsilon = np.linspace(0.9, 0.1, num=max_iteration)[i_ep]
        while True:
            env.render()
            action = agent.sample_action(observation)
            next_observation, reward, done, info, _ = env.step(action)
            #print(observation, action, reward, next_observation, done)
            reward_sum += reward
            agent.update(observation, action, reward, next_observation, done)
            observation = next_observation
            if done:
                print(agent.get_Q())
                print("episode {} reward {}".format(i_ep, reward_sum))
                break
    env.close()
    print(agent.get_Q())
    print(rewards)

