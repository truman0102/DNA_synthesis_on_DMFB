from collections import deque, namedtuple
import numpy as np
import random


class ReplayMemory:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.transition = namedtuple(
            "Transition", ("state", "action", "reward", "next_state", "done")
        )

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) == self.capacity:
            self.memory.popleft()
        self.memory.append(self.transition(state, action, reward, next_state, done))

    def push_batch(self, batch):
        if len(self.memory) + len(batch) > self.capacity:
            self.memory = self.memory[len(batch) :]
        for t in batch:
            self.push(self.transition(*t))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        batch = self.transition(*zip(*batch))
        states = np.array(batch.state)
        actions = np.array(batch.action)
        rewards = np.array(batch.reward)
        next_states = np.array(batch.next_state)
        dones = np.array(batch.done)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


if __name__ == "__main__":
    memory = ReplayMemory(100)
    import numpy as np

    for i in range(40):
        obs = np.random.rand(10, 15, 15)
        action = np.random.randint(0, 5)
        reward = np.random.rand()
        next_obs = np.random.rand(10, 15, 15)
        done = np.random.randint(0, 2)
        memory.push(obs, action, reward, next_obs, done)

    states, actions, rewards, next_states, dones = memory.sample(10)
    print(states.shape)
