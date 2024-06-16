import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ContinuousTask:
    def __init__(self, task_id: int, data: np.ndarray, target: int):
        self.task_id = task_id
        self.data = data
        self.target = target


class ContinuousLearningEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(self, num_tasks=100):
        self.num_tasks = num_tasks
        self.current_step = 0
        self.tasks = self._generate_tasks()

        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(28, 28), dtype=np.float32
        )
        self.action_space = spaces.Discrete(10)

    def _generate_tasks(self):
        tasks = []
        for i in range(self.num_tasks):
            data = np.random.rand(28, 28).astype(np.float32)  # Random image data
            target = np.random.randint(0, 10)  # Random target label
            tasks.append(ContinuousTask(i, data, target))
        return tasks

    def reset(self):
        self.current_step = 0
        self.tasks = self._generate_tasks()
        return self._get_obs()

    def _get_obs(self):
        # Observation is the current task's data
        return self.tasks[self.current_step].data

    def step(self, action):
        task = self.tasks[self.current_step]
        reward = (
            1 if action == task.target else -1
        )  # Reward is 1 for correct, -1 for incorrect
        self.current_step += 1
        done = self.current_step >= self.num_tasks
        info = {"target": task.target}
        return self._get_obs(), reward, done, info

    def render(self, mode="human"):
        pass  # No rendering needed

    def close(self):
        pass  # No resources to clean up


# Register the environment
from gym.envs.registration import register

register(
    id="ContinuousLearningEnv-v0",
    entry_point="gym_examples.envs:ContinuousLearningEnv",
    max_episode_steps=100,
)


# Dummy model for continuous learning
class IncrementalModel(nn.Module):
    def __init__(self):
        super(IncrementalModel, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


def continuous_train(env, model, optimizer, n_epochs=1):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        state = env.reset()
        state = torch.tensor(state).unsqueeze(0)
        done = False
        while not done:
            action = model(state).argmax(dim=1).item()
            next_state, reward, done, info = env.step(action)
            next_state = torch.tensor(next_state).unsqueeze(0)
            target = torch.tensor([info["target"]])

            optimizer.zero_grad()
            output = model(state)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            state = next_state


if __name__ == "__main__":
    env = gym.make("ContinuousLearningEnv-v0")
    model = IncrementalModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Train the model incrementally
    for _ in range(10):  # Simulate continuous learning by training in small epochs
        continuous_train(env, model, optimizer, n_epochs=1)
