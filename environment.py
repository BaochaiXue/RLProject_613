import json
import gym
from gym import spaces
import torch
import threading
import subprocess
import pandas as pd
import numpy as np
import typing
from typing import Dict, List, Any


class DLSchedulingEnv(gym.Env):
    def __init__(self, config_file: str, model_info_file: str):
        super(DLSchedulingEnv, self).__init__()

        # Parse configuration file
        with open(config_file, "r") as file:
            self.config: Dict[str, Any] = json.load(file)

        # Read model information from CSV file
        self.model_info: pd.DataFrame = pd.read_csv(model_info_file)

        self.setup_gpu_streams()
        self.initialize_parameters()
        self.lock: threading.Lock = threading.Lock()
        self.event: threading.Event = threading.Event()

    def setup_gpu_streams(self) -> None:
        # Set up two backend GPU streams with different priorities
        self.stream_high_priority: torch.cuda.Stream = torch.cuda.Stream(priority=0)
        self.stream_low_priority: torch.cuda.Stream = torch.cuda.Stream(priority=1)

    def initialize_parameters(self) -> None:
        # Extract necessary parameters from the config
        self.num_tasks: int = self.config["num_tasks"]
        self.num_variants: int = self.config["num_variants"]
        self.task_list: List[Dict[str, Any]] = self.config["task_list"]
        self.gpu_resources: List[float] = self.config["gpu_resources"]

        # Initialize variant runtimes from model information
        self.variant_runtimes: np.ndarray = self._extract_variant_runtimes()

    def _extract_variant_runtimes(self) -> np.ndarray:
        # Create a dictionary to store the runtimes based on task and variant
        runtime_dict: Dict[str, List[typing.Optional[float]]] = {}
        for _, row in self.model_info.iterrows():
            task_name: str = row["Model Name"]
            variant_id: int = (
                int(row["Model Number"]) - 1
            )  # Assuming variant_id starts from 0
            runtime: float = row["Inference Time (s)"]
            if task_name not in runtime_dict:
                runtime_dict[task_name] = [None] * self.num_variants
            runtime_dict[task_name][variant_id] = runtime

        # Convert the dictionary to a 2D numpy array for use in observation space
        runtimes: List[List[float]] = []
        for task in self.task_list:
            task_name: str = task["name"]
            runtimes.append(runtime_dict[task_name])
        return np.array(runtimes)

    def define_spaces(self) -> None:
        # Define action space (e.g., choosing the next task and task variant)
        self.action_space: spaces.Dict = spaces.Dict(
            {
                "task_id": spaces.Discrete(self.num_tasks),
                "variant_id": spaces.Discrete(self.num_variants),
            }
        )

        # Define observation space (e.g., task deadlines, expected runtimes of each variant, and GPU stream status)
        self.observation_space: spaces.Dict = spaces.Dict(
            {
                "task_deadlines": spaces.Box(
                    low=0, high=float("inf"), shape=(self.num_tasks,), dtype=float
                ),
                "variant_runtimes": spaces.Box(
                    low=0,
                    high=float("inf"),
                    shape=(self.num_tasks, self.num_variants),
                    dtype=float,
                ),
                "gpu_resources": spaces.Box(low=0, high=1, shape=(2,), dtype=float),
                "current_stream": spaces.Discrete(
                    2
                ),  # 0 for high priority, 1 for low priority
            }
        )

    def reset(self) -> Dict[str, Any]:
        self.current_task_status: List[int] = [0] * self.num_tasks
        self.available_gpu_resources: List[float] = [
            1.0,
            1.0,
        ]  # Both GPU streams fully available

        initial_observation: Dict[str, Any] = {
            "task_deadlines": self.config["task_deadlines"],
            "variant_runtimes": self.variant_runtimes,
            "gpu_resources": self.available_gpu_resources,
            "current_stream": 0,  # Starting with the high priority stream
        }
        return initial_observation

    def step(
        self, action: Dict[str, int]
    ) -> typing.Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        task_id: int = action["task_id"]
        variant_id: int = action["variant_id"]

        # Execute the chosen DL task on the appropriate GPU stream
        task: Dict[str, Any] = self.task_list[task_id]
        variant: Dict[str, Any] = task["variants"][variant_id]

        if (
            variant["priority"] == 0
            and self.available_gpu_resources[0] >= variant["resource_demand"]
        ):
            threading.Thread(
                target=self.execute_task,
                args=(task, variant, self.stream_high_priority),
            ).start()
        elif (
            variant["priority"] == 1
            and self.available_gpu_resources[1] >= variant["resource_demand"]
        ):
            threading.Thread(
                target=self.execute_task, args=(task, variant, self.stream_low_priority)
            ).start()
        else:
            # Handle case where resources are insufficient
            reward: float = -1  # Negative reward for invalid action
            done: bool = False
            new_observation: Dict[str, Any] = {
                "task_deadlines": self.config["task_deadlines"],
                "variant_runtimes": self.variant_runtimes,
                "gpu_resources": self.available_gpu_resources,
                "current_stream": 0,
            }
            return new_observation, reward, done, {}

        # Wait for an event-driven call
        self.event.wait()
        self.event.clear()

        # Update task status and calculate reward
        self.current_task_status[task_id] = 1
        reward = self.calculate_reward(task)

        # Check if episode has ended
        done = all(status == 1 for status in self.current_task_status)

        # Prepare new observation
        new_observation = {
            "task_deadlines": self.config["task_deadlines"],
            "variant_runtimes": self.variant_runtimes,
            "gpu_resources": self.available_gpu_resources,
            "current_stream": 0,
        }
        return new_observation, reward, done, {}

    def execute_task(
        self, task: Dict[str, Any], variant: Dict[str, Any], stream: torch.cuda.Stream
    ) -> None:
        command: List[str] = [
            "python",
            "run_task.py",
            task["name"],
            variant["name"],
            str(stream.priority),
        ]
        process: subprocess.Popen = subprocess.Popen(command)
        process.wait()  # Wait for the subprocess to complete

        with self.lock:
            self.update_gpu_resources(variant, stream)

        self.event.set()

    def update_gpu_resources(
        self, variant: Dict[str, Any], stream: torch.cuda.Stream
    ) -> None:
        if stream == self.stream_high_priority:
            self.available_gpu_resources[0] += variant["resource_demand"]
        else:
            self.available_gpu_resources[1] += variant["resource_demand"]

    def calculate_reward(self, task: Dict[str, Any]) -> float:
        # Placeholder reward calculation
        return task["reward"]

    def render(self, mode: str = "human") -> None:
        print(f"Task Status: {self.current_task_status}")
        print(f"GPU Resources: {self.available_gpu_resources}")

    def close(self) -> None:
        # Clean up GPU resources
        del self.stream_high_priority
        del self.stream_low_priority
