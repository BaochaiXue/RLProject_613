import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Any
import random
import time
import subprocess


class TaskSchedulingEnv(gym.Env):
    """
    Custom Environment for RL-based task scheduling with real-time consideration and deadlines.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        num_models: int,
        periodic_tasks: Dict[str, float],
        uncertain_tasks: Dict[str, float],
    ):
        super(TaskSchedulingEnv, self).__init__()

        self.num_models = num_models
        self.periodic_tasks = periodic_tasks
        self.uncertain_tasks = uncertain_tasks

        # Define action and observation space
        # Actions: choose which model to run the inference task on
        self.action_space = spaces.Discrete(num_models)

        # Observations: state of the environment (task queue length for each model)
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(num_models,), dtype=np.float32
        )

        self.reset()

    def reset(self):
        """
        Reset the state of the environment to the initial state.
        """
        self.task_queue = {model_name: [] for model_name in self.periodic_tasks.keys()}
        self.current_time = time.time()

        # Initialize the task queue with periodic tasks
        for model_name, period in self.periodic_tasks.items():
            next_time = self.current_time + period
            ddl = (
                next_time + period
            )  # Set the deadline as twice the period for simplicity
            self.task_queue[model_name].append((next_time, ddl))

        # Initial observation
        self.state = self._get_state()
        return self.state

    def _get_state(self):
        """
        Get the current state of the environment.
        """
        state = np.array(
            [
                len(self.task_queue[model_name])
                for model_name in self.periodic_tasks.keys()
            ]
        )
        return state

    def step(self, action: int):
        """
        Execute one time step within the environment.
        """
        # Update the current time
        self.current_time = time.time()

        # Check for periodic tasks and update task queue
        for model_name, period in self.periodic_tasks.items():
            while (
                self.task_queue[model_name]
                and self.current_time >= self.task_queue[model_name][0][0]
            ):
                self.task_queue[model_name].pop(0)
                next_time = self.current_time + period
                ddl = (
                    next_time + period
                )  # Set the deadline as twice the period for simplicity
                self.task_queue[model_name].append((next_time, ddl))

        # Generate random tasks based on Poisson distribution for uncertain tasks
        for model_name, lambda_val in self.uncertain_tasks.items():
            if np.random.poisson(lambda_val) > 0:
                ddl = self.current_time + random.randint(1, 10)  # Random deadline
                self.task_queue[model_name].append((self.current_time, ddl))

        # Perform the action (inference on the selected model)
        reward = 0
        model_name = list(self.periodic_tasks.keys())[action]
        if self.task_queue[model_name]:
            task_start_time, ddl = self.task_queue[model_name].pop(
                0
            )  # Remove the task as it is being processed

            # Check if the task meets the deadline
            if self.current_time <= ddl:
                # Call the DL_tasks.py script
                result = self._call_dl_tasks_script(model_name, action)
                reward = (
                    1 if result == 0 else -1
                )  # Positive reward if inference is correct, otherwise negative
            else:
                reward = -1  # Negative reward if the task misses the deadline
        else:
            reward = -1  # Negative reward for incorrect inference

        # Update state
        self.state = self._get_state()

        done = False
        if len(self.task_queue) == 0 and all(
            len(tasks) == 0 for tasks in self.task_queue.values()
        ):
            done = True

        return self.state, reward, done, {}

    def _call_dl_tasks_script(self, model_name: str, model_number: int) -> int:
        """
        Call the DL_tasks.py script and return its exit code.
        """
        try:
            result = subprocess.run(
                ["python", "DL_tasks.py", model_name, str(model_number)], check=True
            )
            return result.returncode
        except subprocess.CalledProcessError as e:
            print(f"Subprocess failed with error: {e}")
            return 1  # Assume failure if there is an error in the subprocess call

    def render(self, mode="human", close=False):
        """
        Render the environment to the screen (optional).
        """
        print(
            f"Time: {self.current_time}, Task Queue: {self.task_queue}, State: {self.state}"
        )


# Example usage
if __name__ == "__main__":
    num_models: int = 6
    # Define the periodic tasks for each model (period in seconds)
    periodic_tasks: Dict[str, float] = {
        "vit_b_16": 0.12,
        "resnet50": 0.05,
        "vgg16": 0.08,
        "mobilenet_v3_large": 0.05,
    }
    # Define the uncertain tasks with their Poisson lambda values
    uncertain_tasks: Dict[str, float] = {
        "vit_b_16": 2,
        "resnet50": 2,
    }

    env = TaskSchedulingEnv(
        num_models=num_models,
        periodic_tasks=periodic_tasks,
        uncertain_tasks=uncertain_tasks,
    )

    state = env.reset()
    for _ in range(100):  # Run for 100 steps
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        env.render()
        if done:
            break
