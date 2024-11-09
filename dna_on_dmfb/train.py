import argparse
from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
import os
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

faulthandler.enable()

GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 10000


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = RoutingDMFB()
    memory = ReplayMemory(args.capacity)
    n_actions = env.action_space.n
    policy = DQN().to(device)
    target = deepcopy(policy)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    criterion = nn.MSELoss()
    # optimizer_contrastive = optim.Adam(policy.parameters())
    steps_done = 0
    l_losses = []
    l_rewards = []
    l_steps = []
    results = {}
    name = f"{args.select_action}_loss_reward_steps_lr_{args.max_episodes}_{args.max_steps}_{args.target_update}_{args.lr}"
    time_str = time.strftime("%Y%m%d%H%M%S")
    vis_path = os.path.join("figures/", name, time_str)
    os.makedirs(vis_path, exist_ok=True)
    ckpt_path = os.path.join("checkpoints/", name, time_str)
    os.makedirs(ckpt_path, exist_ok=True)
    for episode in trange(1, args.max_episodes + 1, desc="Episodes", leave=True):
        # for episode in range(1, MAX_EPISODES + 1):
        env.reset()
        # print(f"Episode {episode}")
        losses_episode = []
        rewards_episode = []
        for t in trange(1, args.max_steps + 1, desc="Steps", leave=False):
            step_loss = torch.tensor(0.0).to(device)
            # for t in range(1, MAX_STEPS + 1):
            epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(
                -1.0 * steps_done / EPS_DECAY
            )
            steps_done += 1
            droplet_name, obs = env.select()
            valid_actions = env.valid_actions(droplet_name)
            if len(valid_actions) == 0 or random.random() < epsilon:
                # 随机选择一个有效动作
                action = random.choice(valid_actions)
            else:
                q_values = policy(torch.from_numpy(obs).float().unsqueeze(0).to(device))
                # 正负样本InfoNCE Loss & 选择valid_actions中的最大值
                q_values = q_values.squeeze()
                if args.select_action == "max":
                    action = q_values.clone().detach().argmax().item()
                elif args.select_action == "valid":
                    q_values_for_action = q_values.clone().detach()
                    q_values_for_action[
                        [i for i in range(n_actions) if i not in valid_actions]
                    ] = -np.inf
                    action = q_values_for_action.argmax().item()
                else:
                    raise ValueError("Invalid select action method")
                # print(
                #     f"select action: {action} from {q_values.tolist()} with valid_actions: {valid_actions}"
                # )
                if len(valid_actions) < n_actions:
                    positive_values = q_values[valid_actions]
                    negative_values = q_values[
                        [i for i in range(n_actions) if i not in valid_actions]
                    ]
                    contrastive_loss = -torch.log(
                        torch.exp(positive_values).sum()
                        / (
                            torch.exp(positive_values).sum()
                            + torch.exp(negative_values).sum()
                        )
                    )
                    # optimizer_contrastive.zero_grad()
                    # contrastive_loss.backward()
                    # optimizer_contrastive.step()

                    step_loss += contrastive_loss
            next_obs, reward, terminated, truncated, info = env.step(
                droplet_name, action
            )
            rewards_episode.append(reward)
            done = terminated or truncated
            memory.push(obs, action, reward, next_obs, done)
            # 优化模型

            if len(memory) < args.batch_size:
                continue
            states, actions, rewards, next_states, dones = memory.sample(
                args.batch_size
            )
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
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            step_loss += loss
            optimizer.zero_grad()
            step_loss.backward()
            optimizer.step()

            losses_episode.append(step_loss.item())
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
            if t == args.max_steps:
                print("Steps limit reached")
                results[episode] = {
                    "steps": t,
                    "reward": np.mean(rewards_episode),
                    "loss": np.mean(losses_episode),
                    "success": False,
                }
        if episode % args.target_update == 0:
            target.load_state_dict(policy.state_dict())
        l_losses.append(np.mean(losses_episode))
        l_rewards.append(np.mean(rewards_episode))
        l_steps.append(t)
        scheduler.step()
        env.save_figure(os.path.join(vis_path, f"{episode}.png"))
    print(f"Training done with {len(results)} episodes")
    success = len([1 for episode, result in results.items() if result["success"]])
    print(f"Training done with {success/len(results)*100}% success rate")

    # 保存模型
    torch.save(
        policy.state_dict(),
        os.path.join(ckpt_path, "policy.pth"),
    )
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
        os.path.join(vis_path, "training.png"),
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=15000)
    parser.add_argument("--capacity", type=int, default=20000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.1)
    parser.add_argument("--eps_decay", type=int, default=10000)
    parser.add_argument("--target_update", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--select_action", type=str, default="valid", choices=["max", "valid"]
    )
    args = parser.parse_args()
    train(args)