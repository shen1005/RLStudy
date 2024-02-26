import os

import torch
import matplotlib.pyplot as plt

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)
from PGAgent import PGAgent

def train(env_name, seed, num_episodes, log_dir='experiments/limit-holdem/PG'):
    money = []
    device = get_device()
    set_seed(seed)
    env = rlcard.make(
        env_name,
        config={
            'seed': seed,
        }
    )
    agent = PGAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        hidden_layers=[64, 64],
        device=device,
    )
    # agent = torch.load('experiments/limit-holdem/PG/model.pth')
    agents = [agent]
    for _ in range(1, env.num_players):
        # agents.append(RandomAgent(num_actions=env.num_actions))
        agents.append(torch.load('experiments/limit-holdem/model.pth'))
    env.set_agents(agents)
    with Logger(log_dir) as logger:
        for episode in range(num_episodes):
            if episode == 3:
                print("debug")
            trajectories, payoffs = env.run(is_training=True)
            print(payoffs)
            money.append(payoffs[0])
            trajectories = reorganize(trajectories, payoffs)
            # 可能出现1号选手直接放弃的情况，此时trajectories[0]为空
            if len(trajectories[0]) == 0:
                continue
            states = [ts[0]['obs'] for ts in trajectories[0]]
            actions = [ts[1] for ts in trajectories[0]]
            rewards = [ts[2] for ts in trajectories[0]]
            agent.learn(states, actions, rewards)
            print(f"Episode {episode} finished")
    save_path = os.path.join(log_dir, 'model.pth')
    torch.save(agent, save_path)
    print('Model saved in', save_path)
    plt.plot(money)
    plt.show()

def evaluate(env_name, seed, num_episodes, is_random=False, log_dir='experiments/limit-holdem/PG', model='Ran'):
    device = get_device()
    set_seed(seed)
    env = rlcard.make(
        env_name,
        config={
            'seed': seed,
        }
    )
    if not is_random:
        agent = torch.load(os.path.join(log_dir, 'model.pth'))
    else:
        agent = PGAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            hidden_layers=[64, 64],
            device=device,
        )
    agents = [agent]
    for _ in range(1, env.num_players):
        if model == 'Ran':
            agents.append(RandomAgent(num_actions=env.num_actions))
        else:
            agents.append(torch.load('experiments/limit-holdem/model.pth'))
    env.set_agents(agents)
    rewards = []
    for i in range(1000):
        trajectories, payoffs = env.run(is_training=False)
        rewards.append(payoffs[0])
    # 计算获胜率
    print(f"获胜率: {sum([1 for r in rewards if r > 0]) / 1000}")
    print(f"平均收益: {sum(rewards) / 1000}")

# PG和DQN对打
def against(env_name, seed, num_episodes):
    device = get_device()
    set_seed(seed)
    env = rlcard.make(
        env_name,
        config={
            'seed': seed,
        }
    )
    pg_agent = torch.load('experiments/limit-holdem/PG/model.pth')
    dqn_agent = torch.load('experiments/limit-holdem/model.pth')
    agents = [pg_agent, dqn_agent]
    env.set_agents(agents)
    rewards = []
    for i in range(1000):
        trajectories, payoffs = env.run(is_training=False)
        rewards.append(payoffs[0])
    # 计算获胜率
    print(f"pg获胜率: {sum([1 for r in rewards if r > 0]) / 1000}")
    print(f"pg平均收益: {sum(rewards) / 1000}")

if __name__ == "__main__":
    # against('limit-holdem', 0, 1000)
    evaluate('limit-holdem', 0, 1000, is_random=True, model="DQN")
    train('limit-holdem', 0, 5000)
    evaluate('limit-holdem', 0, 1000)
    against('limit-holdem', 0, 1000)