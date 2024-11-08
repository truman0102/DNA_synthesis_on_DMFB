import argparse
from dna_on_dmfb.enviroment.routing_world import RoutingDMFB
from matplotlib import pyplot as plt
import random
import seaborn as sns
from tqdm import trange


def load_env(args):
    env = RoutingDMFB(
        # size=args.size,
        # situation=situation,
        show_layout=False
    )
    env.show()
    return env


def test_move(env: RoutingDMFB):
    rewards = []
    for step in trange(10000):
        droplet_name, obs = env.select()
        droplet_id = env.droplets_idx[droplet_name]
        droplet = env.droplets[droplet_id]
        valid_actions = env.valid_actions(droplet_name)
        best_action = random.choice(droplet.approaching_actions)
        action = (
            best_action
            if best_action in valid_actions
            else random.choice(valid_actions)
        )
        next_obs, reward, terminated, truncated, info = env.step(droplet_name, action)
        rewards.append(reward)
        obs = next_obs
        done = terminated or truncated
        if done:
            if terminated and info["reason"].__eq__("success"):
                print(f"Success in {step} steps")
            elif terminated and info["reason"].__eq__("steps"):
                print("Steps limit reached")
            elif truncated:
                print(f"{droplet_name} takes an invalid action")
            break
    env.show()
    print(obs.shape)
    # show_obs(observation=obs)
    sns.histplot(rewards, kde=True)
    plt.xlim(-2, 2)
    plt.show()
    plt.close()


def show_obs(observation):
    for o in observation:
        plt.imshow(o, cmap="gray")
        plt.show()
        plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=None)
    args = parser.parse_args()

    env = load_env(args)

    test_move(env)
