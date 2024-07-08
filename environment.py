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
import time
import GPUtil


class DLSchedulingEnv(gym.Env):
    def __init__(self, config_file: str, model_info_file: str):
        super(DLSchedulingEnv, self).__init__()

        # Parse configuration file
        with open(config_file, "r") as file:
            self.config: Dict[str, Any] = json.load(file)

        # Read model information from CSV file
        self.model_info: pd.DataFrame = pd.read_csv(model_info_file)

        self.initialize_parameters()
        self.lock: threading.Lock = threading.Lock()
        self.event: threading.Event = threading.Event()
        self.task_threads: Dict[str, threading.Thread] = {}

        self.define_spaces()
        self.task_queues: Dict[int, List[Dict[str, Any]]] = self.generate_task_queues()

        # Start monitoring GPU streams
        self.monitor_thread: threading.Thread = threading.Thread(
            target=self.monitor_gpu_streams
        )
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def generate_task_queues(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        Generate task queues based on the task list and the total time.
        Each task in the queue has a dictionary with the following keys: start_time and deadline.
        """
        task_queues: Dict[int, List[Dict[str, Any]]] = {}
        current_time: int = 0
        for task_id, task in enumerate(self.task_list):
            task_queue: List[Dict[str, Any]] = []
            if task.get("if_periodic", False):
                period_ms: int = task["period_ms"]
                while current_time < self.total_time_ms:
                    task_queue.append(
                        {
                            "start_time": current_time,
                            "deadline": current_time + period_ms,
                        }
                    )
                    current_time += period_ms
            else:
                possion_lambda: int = task["possion_lambda"]
                while current_time < self.total_time_ms:
                    inter_arrival_time: int = np.random.poisson(possion_lambda)
                    task_queue.append(
                        {
                            "start_time": current_time,
                            "deadline": current_time + inter_arrival_time,
                        }
                    )
                    current_time += inter_arrival_time
            task_queues[task_id] = task_queue
        return task_queues

    def initialize_parameters(self) -> None:
        """
        Extract necessary parameters from the config and initialize variant runtimes and accuracies from model information.
        """
        self.num_tasks: int = self.config["num_tasks"]
        self.num_variants: int = self.config["num_variants"]
        self.total_time_ms: int = self.config["total_time_ms"]
        self.task_list: List[Dict[str, Any]] = self.config["task_list"]

        self.variant_runtimes, self.variant_accuracies = self._extract_variant_info()

    def _extract_variant_info(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Extract runtime and accuracy information for each variant from the model information.
        """
        runtime_dict: Dict[str, List[typing.Optional[float]]] = {}
        accuracy_dict: Dict[str, List[typing.Optional[float]]] = {}
        for _, row in self.model_info.iterrows():
            model_name: str = row["Model Name"]
            variant_id: int = (
                int(row["Model Number"]) - 1
            )  # Assuming variant_id starts from 0
            runtime: float = row["Inference Time (s)"]
            accuracy: float = row["Accuracy (Percentage)"]
            if model_name not in runtime_dict:
                runtime_dict[model_name] = [None] * self.num_variants
                accuracy_dict[model_name] = [None] * self.num_variants
            runtime_dict[model_name][variant_id] = runtime
            accuracy_dict[model_name][variant_id] = accuracy

        runtimes: List[List[float]] = []
        accuracies: List[List[float]] = []
        for task in self.task_list:
            model_name: str = task["model"]
            runtimes.append(runtime_dict[model_name])
            accuracies.append(accuracy_dict[model_name])
        return np.array(runtimes), np.array(accuracies)

    def define_spaces(self) -> None:
        """
        Define the action and observation spaces for the environment.
        """
        self.action_space: spaces.Dict = spaces.Dict(
            {
                "task1_id": spaces.Discrete(
                    self.num_tasks + 1
                ),  # including the empty action
                "variant1_id": spaces.Discrete(self.num_variants),
                "task2_id": spaces.Discrete(
                    self.num_tasks + 1
                ),  # including the empty action
                "variant2_id": spaces.Discrete(self.num_variants),
            }
        )

        self.observation_space: spaces.Dict = spaces.Dict(
            {
                "current_streams_status": spaces.MultiBinary(
                    2
                ),  # we have 2 streams, true if busy, false if idle
                "current_time": spaces.Box(
                    low=0, high=float("inf"), shape=(1,), dtype=float
                ),
                "task_deadlines": spaces.Box(
                    low=0, high=float("inf"), shape=(self.num_tasks,), dtype=float
                ),
                "task_if_arrived": spaces.Discrete(self.num_tasks),
                "task_if_periodic": spaces.Discrete(self.num_tasks),
                "variant_runtimes": spaces.Box(
                    low=0,
                    high=float("inf"),
                    shape=(self.num_tasks, self.num_variants),
                    dtype=float,
                ),
                "variant_accuracies": spaces.Box(
                    low=0,
                    high=100,
                    shape=(self.num_tasks, self.num_variants),
                    dtype=float,
                ),
                "gpu_resources": spaces.Box(low=0, high=1, shape=(2,), dtype=float),
            }
        )

    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment and return the initial observation.
        """
        self.current_task_status: List[int] = [0] * self.num_tasks
        self.task_start_times: Dict[int, float] = {}
        self.task_end_times: Dict[int, float] = {}
        self.stream_high_priority: torch.cuda.Stream = torch.cuda.Stream(priority=0)
        self.stream_low_priority: torch.cuda.Stream = torch.cuda.Stream(priority=1)

        initial_observation: Dict[str, Any] = {
            "current_streams_status": [False, False],
            "current_time": np.array([0.0]),
            "task_deadlines": np.array([task["deadline"] for task in self.task_list]),
            "task_if_arrived": np.zeros(self.num_tasks),
            "task_if_periodic": np.array(
                [task.get("if_periodic", 0) for task in self.task_list]
            ),
            "variant_runtimes": self.variant_runtimes,
            "variant_accuracies": self.variant_accuracies,
            "gpu_resources": self.get_gpu_resources(),
        }
        return initial_observation

    def step(
        self, action: Dict[str, int]
    ) -> typing.Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute a step in the environment based on the action provided.
        """
        task_id: int = action["task1_id"]
        variant_id: int = action["variant1_id"]

        task: Dict[str, Any] = self.task_list[task_id]
        variant_runtime = self.variant_runtimes[task_id][variant_id]
        variant_accuracy = self.variant_accuracies[task_id][variant_id]

        gpu_resources = self.get_gpu_resources()
        if gpu_resources[0] >= variant_runtime:
            thread = threading.Thread(
                target=self.execute_task,
                args=(
                    task,
                    variant_runtime,
                    variant_accuracy,
                    task_id,
                    variant_id,
                    self.stream_high_priority,
                ),
            )
            thread.start()
            self.task_threads[f"{task_id}_{variant_id}"] = thread
        elif gpu_resources[1] >= variant_runtime:
            thread = threading.Thread(
                target=self.execute_task,
                args=(
                    task,
                    variant_runtime,
                    variant_accuracy,
                    task_id,
                    variant_id,
                    self.stream_low_priority,
                ),
            )
            thread.start()
            self.task_threads[f"{task_id}_{variant_id}"] = thread
        else:
            # Handle case where resources are insufficient
            reward: float = -1  # Negative reward for invalid action
            done: bool = False
            new_observation: Dict[str, Any] = {
                "task_deadlines": [self.config.get("total_time_ms", 10000)]
                * self.num_tasks,
                "variant_runtimes": self.variant_runtimes,
                "variant_accuracies": self.variant_accuracies,
                "gpu_resources": self.get_gpu_resources(),
                "current_stream": 0,
            }
            return new_observation, reward, done, {}

        # Check task status and update resources
        self.check_tasks()

        # Update task status and calculate reward
        self.current_task_status[task_id] = 1
        reward = self.calculate_reward(task_id, variant_id)

        # Check if episode has ended
        done = all(status == 1 for status in self.current_task_status)

        # Prepare new observation
        new_observation = {
            "task_deadlines": [self.config.get("total_time_ms", 10000)]
            * self.num_tasks,
            "variant_runtimes": self.variant_runtimes,
            "variant_accuracies": self.variant_accuracies,
            "gpu_resources": self.get_gpu_resources(),
            "current_stream": 0,
        }
        return new_observation, reward, done, {}

    def execute_task(
        self,
        task: Dict[str, Any],
        variant_runtime: float,
        variant_accuracy: float,
        task_id: int,
        variant_id: int,
        stream: torch.cuda.Stream,
    ) -> None:
        """
        Execute the specified task variant on the given GPU stream.
        """
        command: List[str] = [
            "python",
            task["address"],
            task["data"],
            str(stream.priority),
        ]
        process: subprocess.Popen = subprocess.Popen(command)
        self.task_start_times[task_id] = time.time()
        self.monitor_task(process, variant_runtime, variant_accuracy, task_id, stream)

    def monitor_task(
        self,
        process: subprocess.Popen,
        variant_runtime: float,
        variant_accuracy: float,
        task_id: int,
        stream: torch.cuda.Stream,
    ) -> None:
        """
        Monitor the task execution and update resources upon completion.
        """
        thread = threading.Thread(
            target=self._monitor_task,
            args=(process, variant_runtime, variant_accuracy, task_id, stream),
        )
        thread.start()

    def _monitor_task(
        self,
        process: subprocess.Popen,
        variant_runtime: float,
        variant_accuracy: float,
        task_id: int,
        stream: torch.cuda.Stream,
    ) -> None:
        """
        Wait for the task to complete and update the GPU resources.
        """
        process.wait()  # Wait for the subprocess to complete
        with self.lock:
            self.update_gpu_resources(variant_runtime, stream)
        self.task_end_times[task_id] = time.time()
        self.event.set()

    def get_gpu_resources(self) -> List[float]:
        """
        Get the current GPU load and memory usage.
        """
        gpus = GPUtil.getGPUs()
        if not gpus:
            return [1.0, 1.0]  # Default to fully available if no GPUs are found
        loads = [gpu.load for gpu in gpus[:2]]  # Get load for the first two GPUs
        return [1.0 - load for load in loads]

    def update_gpu_resources(
        self, variant_runtime: float, stream: torch.cuda.Stream
    ) -> None:
        """
        Update the GPU resources after a task completes.
        """
        # Implementation details for updating GPU resources can be added here

    def calculate_reward(self, task_id: int, variant_id: int) -> float:
        """
        Calculate the reward based on task completion, GPU utilization, and accuracy.
        """
        deadline = (
            self.config.get("total_time_ms", 10000) / 1000.0
        )  # Convert to seconds
        end_time = self.task_end_times[task_id]
        start_time = self.task_start_times[task_id]
        completion_time = end_time - start_time if end_time and start_time else 0
        ddl_miss_penalty = -10 if completion_time > deadline else 0

        gpu_utilization = 1.0 - np.mean(self.get_gpu_resources())
        utilization_reward = gpu_utilization * 10

        accuracy = self.variant_accuracies[task_id][variant_id]
        accuracy_reward = accuracy / 10  # Scale the accuracy to a smaller value

        return utilization_reward + accuracy_reward + ddl_miss_penalty

    def render(self, mode: str = "human") -> None:
        """
        Render the current state of the environment.
        """
        print(f"Task Status: {self.current_task_status}")
        print(f"GPU Resources: {self.get_gpu_resources()}")

    def close(self) -> None:
        """
        Clean up the environment and GPU resources.
        """
        del self.stream_high_priority
        del self.stream_low_priority

    def check_tasks(self) -> None:
        """
        Check the status of all tasks and update the event accordingly.
        """
        for key, thread in list(self.task_threads.items()):
            if not thread.is_alive():
                del self.task_threads[key]
                self.event.set()

    def monitor_gpu_streams(self) -> None:
        """
        Continuously monitor the GPU streams for task completion.
        """
        while True:
            self.check_tasks()
            time.sleep(0.1)  # Check every 0.1 seconds


# Example usage:
env = DLSchedulingEnv(
    config_file="config.json", model_info_file="model_information.csv"
)

# Reset environment to get the initial observation
observation = env.reset()


# Simple agent that takes random actions
class SimpleContinuousAgent:
    def __init__(
        self, action_space: spaces.Dict, observation_space: spaces.Dict
    ) -> None:
        self.action_space = action_space
        self.observation_space = observation_space

    def select_action(self, observation: Dict[str, Any]) -> Dict[str, int]:
        """
        Select a random action from the action space.
        """
        task_id = np.random.randint(self.action_space["task1_id"].n)
        variant_id = np.random.randint(self.action_space["variant1_id"].n)
        return {"task1_id": task_id, "variant1_id": variant_id}


agent = SimpleContinuousAgent(env.action_space, env.observation_space)

# Continuous interaction loop
for _ in range(100):  # Run for 100 iterations as an example
    action = agent.select_action(observation)
    observation, reward, done, info = env.step(action)

    print(f"Action: {action}")
    print(f"Observation: {observation}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")

    if done:
        print("Episode finished")
        break

env.close()
