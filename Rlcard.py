import os

import torch
import matplotlib.pyplot as plt

import rlcard
from rlcard.agents import RandomAgent, DQNAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)


def train(env_name, seed, num_episodes, log_dir='experiments/limit-holdem'):
    rewards = []
    device = get_device()
    set_seed(seed)
    env = rlcard.make(
        env_name,
        config={
            'seed': seed,
        }
    )
    agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=[64, 64],
        device=device,
    )
    agents = [agent]
    for _ in range(1, env.num_players):
        agents.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)
    with Logger(log_dir) as logger:
        for episode in range(num_episodes):
            trajectories, payoffs = env.run(is_training=True)
            print(payoffs)
            rewards.append(payoffs[0])
            trajectories = reorganize(trajectories, payoffs)
            for ts in trajectories[0]:
                agent.feed(ts)
    save_path = os.path.join(log_dir, 'model.pth')
    torch.save(agent, save_path)
    print('Model saved in', save_path)
    plt.plot(rewards)
    plt.show()

def evaluate(env_name, seed, num_episodes, log_dir='experiments/limit-holdem'):
    device = get_device()
    set_seed(seed)
    env = rlcard.make(
        env_name,
        config={
            'seed': seed,
        }
    )
    agent = torch.load(os.path.join(log_dir, 'model.pth'))
    agents = [agent]
    for _ in range(1, env.num_players):
        agents.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)
    rewards = []
    for i in range(1000):
        trajectories, payoffs = env.run(is_training=False)
        rewards.append(payoffs[0])
    # 计算获胜率
    print(f"获胜率: {sum([1 for r in rewards if r > 0]) / 1000}")


if __name__ == '__main__':
    envName = 'limit-holdem'
    seed = 42
    num_episodes = 5000
    log_dir = './experiments/limit-holdem'
    train(envName, seed, num_episodes, log_dir=log_dir)
    evaluate(envName, seed, num_episodes, log_dir=log_dir)