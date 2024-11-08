import argparse
from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dna_on_dmfb.enviroment.routing_world import RoutingDMFB
from dna_on_dmfb.models.dqn import DQN
from dna_on_dmfb.utils.rl import ReplayMemory

import faulthandler

# 在import之后直接添加以下启用代码即可
faulthandler.enable()

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 10000
TARGET_UPDATE = 2
MAX_EPISODES = 500
MAX_STEPS = 10000
memory = ReplayMemory(10000)


def epsilon_greedy_action(model, state, epsilon, valid_actions):
    if random.random() < epsilon:
        # 随机选择一个有效动作
        # action = random.choice(list(range(num_actions)))
        action = random.choice(valid_actions)
    else:
        with torch.no_grad():
            q_values = model(state)
            q_values = q_values.squeeze()
            # 只从有效动作中选择Q值最大的动作
            action = q_values.argmax().item()
    return action


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = RoutingDMFB()
    n_actions = env.action_space.n
    policy = DQN().to(device)
    target = deepcopy(policy)
    optimizer = optim.Adam(policy.parameters())
    criterion = nn.MSELoss()

    steps_done = 0
    l_losses = []
    l_rewards = []
    l_steps = []
    results = {}
    for episode in trange(1, MAX_EPISODES + 1, desc="Episodes", leave=True):
        # for episode in range(1, MAX_EPISODES + 1):
        env.reset()
        # print(f"Episode {episode}")
        losses_episode = []
        rewards_episode = []
        for t in trange(1, MAX_STEPS + 1, desc="Steps", leave=False):
            # for t in range(1, MAX_STEPS + 1):
            epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(
                -1.0 * steps_done / EPS_DECAY
            )
            steps_done += 1
            droplet_name, obs = env.select()
            valid_actions = env.valid_actions(droplet_name)
            # obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = epsilon_greedy_action(
                policy,
                torch.from_numpy(obs).float().unsqueeze(0).to(device),
                epsilon,
                valid_actions,
            )
            next_obs, reward, terminated, truncated, info = env.step(
                droplet_name, action
            )
            # next_obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
            rewards_episode.append(reward)
            # reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            memory.push(obs, action, reward, next_obs, done)
            # 优化模型

            if len(memory) < BATCH_SIZE:
                continue
            states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
            states = torch.tensor(states, dtype=torch.float32).to(device)
            actions = torch.tensor(actions, dtype=torch.long).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
            dones = torch.tensor(dones, dtype=torch.float32).to(device)

            q_values = policy(states)
            next_q_values = target(next_states)
            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_value = next_q_values.max(1)[0]
            expected_q_value = rewards + GAMMA * next_q_value * (1 - dones)
            loss = criterion(q_value, expected_q_value.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_episode.append(loss.item())

            if terminated or truncated:
                if terminated and info["reason"].__eq__("success"):
                    print(f"Success in {t} steps")
                    results[episode] = {
                        "steps": t,
                        "reward": np.mean(rewards_episode),
                        "loss": np.mean(losses_episode),
                        "success": True,
                    }
                elif terminated and info["reason"].__eq__("steps"):
                    print("Steps limit reached")
                    results[episode] = {
                        "steps": t,
                        "reward": np.mean(rewards_episode),
                        "loss": np.mean(losses_episode),
                        "success": False,
                    }
                elif truncated:
                    print(f"{droplet_name} takes an invalid action")
                    results[episode] = {
                        "steps": t,
                        "reward": np.mean(rewards_episode),
                        "loss": np.mean(losses_episode),
                        "success": False,
                    }
                break
        if episode % TARGET_UPDATE == 0:
            target.load_state_dict(policy.state_dict())
        l_losses.append(np.mean(losses_episode))
        l_rewards.append(np.mean(rewards_episode))
        l_steps.append(t)
    success = len([1 for episode, result in results.items() if result["success"]])
    print(f"Training done with {success/len(results)*100}% success rate")

    # 绘制训练过程中的步数、损失和奖励
    plt.figure(figsize=(12, 6))
    plt.subplot(131)
    plt.plot(l_losses)
    plt.title("Loss")
    plt.subplot(132)
    plt.plot(l_rewards)
    plt.title("Reward")
    plt.subplot(133)
    plt.plot(l_steps)
    plt.title("Steps")
    plt.savefig(
        f"figures/{MAX_EPISODES}_{MAX_STEPS}_{BATCH_SIZE}_{time.strftime('%Y%m%d%H%M%S')}.png",
        dpi=300,
        bbox_inches="tight",
    )


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    train(args)
