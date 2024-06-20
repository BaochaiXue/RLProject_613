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
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv


class TaskSchedulingEnv(gym.Env):
    """
    Custom Environment for RL-based task scheduling with real-time consideration and deadlines.
    We generate periodic tasks for each model and uncertain tasks based on Poisson distribution, so that we get a predefined task queue.
    When the RL algorithm observes the state of the environment, we will give them the part of the task queue for each model,
    we only show the not done tasks in the task queue, and the start time is earlier than the current time.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        num_models: int,
        periodic_tasks: Dict[str, float],
        uncertain_tasks: Dict[str, float],
        num_model_variants: int,
        model_info: pd.DataFrame,
    ) -> None:
        super(TaskSchedulingEnv, self).__init__()

        self.num_models: int = num_models
        self.periodic_tasks: Dict[str, float] = periodic_tasks
        self.uncertain_tasks: Dict[str, float] = uncertain_tasks
        self.num_model_variants: int = num_model_variants
        self.model_info: pd.DataFrame = model_info

        # Define action and observation space
        # Actions: choose which model to run the inference task on, with the model variant
        self.action_space: spaces.Discrete = spaces.Discrete(
            num_models * num_model_variants
        )

        # Observations: state of the environment (task queue length for each model)
        self.observation_space: spaces.Box = spaces.Box(
            low=0, high=np.inf, shape=(num_models,), dtype=np.float32
        )

        self.current_time: float = 0.0
        # Task queue to store the tasks for each model, with their start time, deadline, predicted_inference_time, and predicted_accuracy
        self.task_queue: Dict[str, List[Tuple[float, float, float, float]]] = {}
        self.state: np.ndarray = np.array([])

        self.reset()

    def reset(self) -> np.ndarray:
        """
        Reset the state of the environment to the initial state.
        """
        self.task_queue = {model_name: [] for model_name in self.periodic_tasks.keys()}
        self.current_time = time.time()

        # Initialize the task queue with 1000 periodic tasks
        for model_name, period in self.periodic_tasks.items():
            for i in range(1000):
                next_time: float = self.current_time + period * (i + 1)
                ddl: float = (
                    next_time + period
                )  # Set the deadline as twice the period for simplicity
                model_info_row: pd.DataFrame = self.model_info[
                    self.model_info["Model Name"] == model_name
                ]
                predicted_inference_time: float = model_info_row[
                    "Inference Time (s)"
                ].values[0]
                predicted_accuracy: float = model_info_row["Accuracy (%)"].values[0]
                self.task_queue[model_name].append(
                    (next_time, ddl, predicted_inference_time, predicted_accuracy)
                )

        # Add uncertain tasks
        for model_name, lambda_val in self.uncertain_tasks.items():
            num_tasks: int = np.random.poisson(lambda_val * 1000)
            for _ in range(num_tasks):
                ddl: float = self.current_time + random.randint(
                    1, 10
                )  # Random deadline
                model_info_row: pd.DataFrame = self.model_info[
                    self.model_info["Model Name"] == model_name
                ]
                predicted_inference_time: float = model_info_row[
                    "Inference Time (s)"
                ].values[0]
                predicted_accuracy: float = model_info_row["Accuracy (%)"].values[0]
                self.task_queue[model_name].append(
                    (
                        self.current_time,
                        ddl,
                        predicted_inference_time,
                        predicted_accuracy,
                    )
                )

        # Initial observation
        self.state = self._get_state()
        return self.state

    def _get_state(self) -> np.ndarray:
        """
        Get the current state of the environment.
        """
        state: np.ndarray = np.array(
            [
                len(self.task_queue[model_name])
                for model_name in self.periodic_tasks.keys()
            ]
        )
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
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
                next_time: float = self.current_time + period
                ddl: float = (
                    next_time + period
                )  # Set the deadline as twice the period for simplicity
                model_info_row: pd.DataFrame = self.model_info[
                    self.model_info["Model Name"] == model_name
                ]
                predicted_inference_time: float = model_info_row[
                    "Inference Time (s)"
                ].values[0]
                predicted_accuracy: float = model_info_row["Accuracy (%)"].values[0]
                self.task_queue[model_name].append(
                    (next_time, ddl, predicted_inference_time, predicted_accuracy)
                )

        # Generate random tasks based on Poisson distribution for uncertain tasks
        for model_name, lambda_val in self.uncertain_tasks.items():
            if np.random.poisson(lambda_val) > 0:
                ddl: float = self.current_time + random.randint(
                    1, 10
                )  # Random deadline
                model_info_row: pd.DataFrame = self.model_info[
                    self.model_info["Model Name"] == model_name
                ]
                predicted_inference_time: float = model_info_row[
                    "Inference Time (s)"
                ].values[0]
                predicted_accuracy: float = model_info_row["Accuracy (%)"].values[0]
                self.task_queue[model_name].append(
                    (
                        self.current_time,
                        ddl,
                        predicted_inference_time,
                        predicted_accuracy,
                    )
                )

        # Perform the action (inference on the selected model)
        reward: float = 0.0
        model_name: str = list(self.periodic_tasks.keys())[
            action % len(self.periodic_tasks)
        ]
        model_variant: int = action // len(self.periodic_tasks)
        if model_variant < self.num_model_variants:
            if self.task_queue[model_name]:
                (
                    task_start_time,
                    ddl,
                    predicted_inference_time,
                    predicted_accuracy,
                ) = self.task_queue[model_name].pop(
                    0
                )  # Remove the task as it is being processed

                # Check if the task meets the deadline
                if self.current_time <= ddl:
                    # Get model file from the dataframe
                    model_info_row: pd.DataFrame = self.model_info[
                        (self.model_info["Model Name"] == model_name)
                        & (self.model_info["Model Number"] == model_variant)
                    ]
                    if not model_info_row.empty:
                        model_file: str = model_info_row["Model File"].values[0]

                        # Call the DL_tasks.py script
                        result: int = self._call_dl_tasks_script(model_file)
                        reward = (
                            1.0 if result == 0 else -1.0
                        )  # Reward based on actual inference result
                    else:
                        reward = -1.0  # Negative reward if model info is not found
                else:
                    reward = -1.0  # Negative reward if the task misses the deadline
            else:
                reward = -1.0  # Negative reward for incorrect inference
        else:
            reward = -1.0  # Negative reward for invalid action

        # Update state
        self.state = self._get_state()

        # Termination condition: task queue is empty
        done: bool = all(len(tasks) == 0 for tasks in self.task_queue.values())

        return self.state, reward, done, {}

    def _call_dl_tasks_script(self, model_file: str) -> int:
        """
        Call the DL_tasks.py script and return its exit code.
        """
        try:
            result: subprocess.CompletedProcess = subprocess.run(
                ["python", "DL_tasks.py", model_file], check=True
            )
            return result.returncode
        except subprocess.CalledProcessError as e:
            print(f"Subprocess failed with error: {e}")
            return 1  # Assume failure if there is an error in the subprocess call

    def render(self, mode: str = "human", close: bool = False) -> None:
        """
        Render the environment to the screen (optional).
        """
        print(
            f"Time: {self.current_time}, Task Queue: {self.task_queue}, State: {self.state}"
        )


# Example usage with real-time functionality
if __name__ == "__main__":
    num_models: int = 4
    num_model_variants: int = 5
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

    # Load the model evaluation information from CSV
    # The columns are: Model Name, Model Number, Model File, Inference Time (s), Accuracy (%)
    model_info: pd.DataFrame = pd.read_csv("model_evaluation.csv")

    env = TaskSchedulingEnv(
        num_models=num_models,
        periodic_tasks=periodic_tasks,
        uncertain_tasks=uncertain_tasks,
        num_model_variants=num_model_variants,
        model_info=model_info,
    )

    # Check the environment
    check_env(env)

    # Create the RL agent
    env = DummyVecEnv([lambda: env])  # Wrap the environment
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the RL agent
    model.learn(total_timesteps=10000)

    # Save the model
    model.save("ppo_task_scheduling")

    # Load the trained model
    model = PPO.load("ppo_task_scheduling")

    # Test the trained model with real-time delay
    obs: np.ndarray = env.reset()
    for _ in range(100):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            break
        time.sleep(1)  # Real-time delay between steps
