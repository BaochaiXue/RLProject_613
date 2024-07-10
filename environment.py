import json
import gym
from gym import spaces
import torch
import concurrent.futures
import subprocess
import pandas as pd
import numpy as np
import typing
from typing import Dict, List, Any, Callable, Tuple
import time
import GPUtil
import os
import random
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch import nn
import threading
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
import sys
import contextlib


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def load_single_test_image(vit_16_using: bool) -> DataLoader:
    """
    Loads a single test image from the CIFAR10 dataset, applies transformations,
    and returns it as a DataLoader object.

    Parameters:
        vit_16_using (bool): Flag indicating whether ViT-16 is being used.

    Returns:
        DataLoader: DataLoader object containing the transformed single test image.
    """
    if vit_16_using:
        transform: transforms.Compose = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                ),
            ]
        )
    else:
        transform: transforms.Compose = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                ),
            ]
        )

    with suppress_stdout():
        testset: datasets.CIFAR10 = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
    random_index: int = random.randint(0, len(testset) - 1)
    test_subset: Subset = Subset(testset, [random_index])
    testloader: DataLoader = DataLoader(test_subset, batch_size=1, shuffle=False)
    return testloader


def load_model(model_name: str, model_number: int) -> nn.Module:
    """
    Loads a pre-trained model from the specified path.

    Parameters:
        model_name (str): Name of the model to be loaded.
        model_number (int): Specific number of the model variant.

    Returns:
        nn.Module: Loaded model.
    """
    model_file: str = f"selected_models/{model_name}/{model_name}_{model_number}.pth"
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file {model_file} not found.")
    model: nn.Module = torch.load(model_file)
    return model


def get_gpu_resources() -> Tuple[float, float]:
    """
    Retrieves the GPU utility rate and GPU memory utility rate of all GPUs.

    Returns:
        Tuple[float, float]: A tuple containing the average GPU utility rate and
                            the average GPU memory utility rate of all GPUs.
    """
    # Retrieve a list of all GPUs
    gpus: List[GPUtil.GPU] = GPUtil.getGPUs()

    # Initialize variables to store the sum of utility rates
    total_gpu_util: float = 0.0
    total_mem_util: float = 0.0

    # Iterate over all available GPUs
    gpu: GPUtil.GGPU
    for gpu in gpus:
        total_gpu_util += gpu.load
        total_mem_util += gpu.memoryUtil

    # Calculate the average utility rates
    num_gpus: int = len(gpus)
    avg_gpu_util: float = total_gpu_util / num_gpus if num_gpus > 0 else 0.0
    avg_mem_util: float = total_mem_util / num_gpus if num_gpus > 0 else 0.0

    return avg_gpu_util, avg_mem_util


class DLSchedulingEnv(gym.Env):
    def __init__(self, config_file: str, model_info_file: str) -> None:
        """
        Initializes the deep learning scheduling environment.

        Parameters:
            config_file (str): Path to the configuration file.
            model_info_file (str): Path to the model information file.
        """
        super(DLSchedulingEnv, self).__init__()

        with open(config_file, "r") as file:
            self.config: Dict[str, Any] = json.load(file)

        self.model_info: pd.DataFrame = pd.read_csv(model_info_file)
        self.start_time: float = time.time() * 1000
        self.initialize_parameters()
        self.define_spaces()
        self.task_queues: Dict[int, List[Dict[str, Any]]] = self.generate_task_queues()
        self.current_task_pointer: Dict[int, int] = {
            task_id: 0 for task_id in range(self.num_tasks)
        }
        self.if_periodic: List[bool] = [
            task.get("if_periodic", False) for task in self.task_list
        ]
        self.stream_status: List[bool] = [
            False,
            False,
        ]  # False means idle, True means busy
        self.task_arrived: List[bool] = [False] * self.num_tasks
        self.executor: concurrent.futures.ThreadPoolExecutor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=2)
        )
        self.lock: threading.Lock = threading.Lock()
        self.total_task_finished: List[int] = [0] * self.num_tasks
        self.total_task_accurate: List[int] = [0] * self.num_tasks
        self.total_missed_deadlines: List[int] = [0] * self.num_tasks
        self.future_to_stream_index: Dict[concurrent.futures.Future, int] = {}
        self.futures: List[concurrent.futures.Future] = []
        self.streams: List[torch.cuda.Stream] = [
            torch.cuda.Stream(priority=-1),
            torch.cuda.Stream(priority=0),
        ]

    def generate_task_queues(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        Generates task queues for all tasks based on their configuration.

        Returns:
            Dict[int, List[Dict[str, Any]]]: Dictionary containing task queues for each task.
        """
        task_queues: Dict[int, List[Dict[str, Any]]] = {}
        for task_id, task in enumerate(self.task_list):
            task_queue: List[Dict[str, Any]] = []
            current_time: int = 0
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
                possion_lambda: float = task["possion_lambda"]
                while current_time < self.total_time_ms:
                    inter_arrival_time: int = np.random.poisson(possion_lambda)
                    task_queue.append(
                        {
                            "start_time": current_time,
                            "deadline": current_time + task["deadline_ms"],
                        }
                    )
                    current_time += inter_arrival_time
            task_queues[task_id] = task_queue
        return task_queues

    def initialize_parameters(self) -> None:
        """
        Initializes the parameters for the environment based on the configuration file.
        """
        self.num_tasks: int = self.config["num_tasks"]
        self.num_variants: int = self.config["num_variants"]
        self.total_time_ms: int = self.config["total_time_ms"]
        self.task_list: List[Dict[str, Any]] = self.config["task_list"]

        self.variant_runtimes, self.variant_accuracies = self._extract_variant_info()

    def _extract_variant_info(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts variant runtime and accuracy information from the model information file.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing numpy arrays of variant runtimes and accuracies.
        """
        runtime_dict: Dict[str, List[float]] = {}
        accuracy_dict: Dict[str, List[float]] = {}
        for _, row in self.model_info.iterrows():
            model_name: str = row["Model Name"]
            variant_id: int = int(row["Model Number"]) - 1
            runtime: float = row["Inference Time (s)"] * 1000
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
        Defines the action and observation spaces for the environment.
        """
        self.action_space: spaces.MultiDiscrete = spaces.MultiDiscrete(
            [
                self.num_tasks + 1,
                self.num_variants,
                self.num_tasks + 1,
                self.num_variants,
            ]
        )

        self.observation_space: spaces.Dict = spaces.Dict(
            {
                "current_streams_status": spaces.MultiBinary(2),
                "current_time": spaces.Box(
                    low=0, high=float("inf"), shape=(1,), dtype=np.float32
                ),
                "task_deadlines": spaces.Box(
                    low=0, high=float("inf"), shape=(self.num_tasks,), dtype=np.float32
                ),
                "task_if_arrived": spaces.MultiBinary(self.num_tasks),
                "task_if_periodic": spaces.MultiBinary(self.num_tasks),
                "variant_runtimes": spaces.Box(
                    low=0,
                    high=float("inf"),
                    shape=(self.num_tasks, self.num_variants),
                    dtype=np.float32,
                ),
                "variant_accuracies": spaces.Box(
                    low=0,
                    high=100,
                    shape=(self.num_tasks, self.num_variants),
                    dtype=np.float32,
                ),
                "gpu_resources": spaces.Box(
                    low=0, high=1, shape=(2,), dtype=np.float32
                ),
            }
        )

    def reset(self) -> Dict[str, Any]:
        """
        Resets the environment to its initial state.

        Returns:
            Dict[str, Any]: Initial observation of the environment.
        """
        self.current_task_pointer = {task_id: 0 for task_id in range(self.num_tasks)}
        self.task_start_times: Dict[int, float] = {}
        self.task_end_times: Dict[int, float] = {}
        self.start_time = time.time() * 1000
        task_if_arrived = np.zeros(self.num_tasks, dtype=np.float32)
        current_time_ms: float = time.time() * 1000
        for task_id, queue in self.task_queues.items():
            if self.current_task_pointer[task_id] < len(queue):
                if_task_available: bool = False
                while (
                    self.current_task_pointer[task_id] < len(queue)
                    and queue[self.current_task_pointer[task_id]]["deadline"]
                    <= current_time_ms - self.start_time
                ):
                    self.current_task_pointer[task_id] += 1
                if (
                    self.current_task_pointer[task_id] < len(queue)
                    and queue[self.current_task_pointer[task_id]]["start_time"]
                    <= current_time_ms - self.start_time
                ):
                    task_if_arrived[task_id] = 1
                    if_task_available = True
                if not if_task_available:
                    task_if_arrived[task_id] = 0
            else:
                task_if_arrived[task_id] = 0
        self.task_arrived = task_if_arrived
        initial_observation: Dict[str, Any] = {
            "current_streams_status": np.array([0, 0], dtype=np.float32),
            "current_time": np.array(
                [current_time_ms - self.start_time], dtype=np.float32
            ),
            "task_deadlines": np.array(
                [
                    (
                        queue[self.current_task_pointer[task_id]]["deadline"]
                        - (current_time_ms - self.start_time)
                        if self.current_task_pointer[task_id] < len(queue)
                        else -float("inf")
                    )
                    for task_id, queue in self.task_queues.items()
                ],
                dtype=np.float32,
            ),
            "task_if_arrived": task_if_arrived,
            "task_if_periodic": np.array(self.if_periodic, dtype=np.float32),
            "variant_runtimes": self.variant_runtimes.astype(np.float32),
            "variant_accuracies": self.variant_accuracies.astype(np.float32),
            "gpu_resources": np.array(get_gpu_resources(), dtype=np.float32),
        }

        return initial_observation

    def execute_task(
        self,
        task: Dict[str, Any],
        task_id: int,
        variant_id: int,
        stream_index: int,  # 0 or 1
        deadline: float,
    ) -> Tuple[int, int]:
        """
        Executes a given task on a specified stream.

        Parameters:
            task (Dict[str, Any]): The task to be executed.
            task_id (int): The ID of the task.
            variant_id (int): The variant ID of the task.
            stream_index (int): The index of the stream (0 or 1).
            deadline (float): The deadline for the task.

        Returns:
            Tuple[int, int]: Result of the task execution indicating correctness and penalty.
        """
        model: nn.Module = load_model(task["model"], variant_id + 1)
        dataloader: DataLoader = load_single_test_image("vit" in task["model"])
        device: torch.device = torch.device(
            f"cuda:0" if torch.cuda.is_available() else "cpu"
        )
        stream: torch.cuda.Stream = self.streams[stream_index]
        self.current_task_pointer[task_id] += 1
        self.task_arrived[task_id] = False
        penalty_function: Callable[[float], float] = lambda x: x * 10 if x > 0 else x

        with torch.cuda.stream(stream):
            model.eval()
            correct: int = 0
            model.to(device)
            with torch.no_grad():
                for images, labels in dataloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
            finish_time: float = time.time() * 1000
            # Update the total task finished and accuracy counts
            with self.lock:
                self.total_task_finished[task_id] += 1
                if correct == 1:
                    self.total_task_accurate[task_id] += 1
                if finish_time - self.start_time > deadline:
                    self.total_missed_deadlines[task_id] += 1
        return (
            correct,
            penalty_function(finish_time - self.start_time - deadline),
        )

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Executes a step in the environment based on the provided action.

        Parameters:
            action (np.ndarray): Array containing the actions to be performed.

        Returns:
            Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
                - Observation after performing the action.
                - Reward obtained.
                - Boolean indicating if the episode is done.
                - Additional information.
        """
        task1_id, variant1_id, task2_id, variant2_id = action

        if_first_action_is_idle: bool = task1_id == self.num_tasks
        if_second_action_is_idle: bool = task2_id == self.num_tasks
        tmp_reward: float = 0.0
        # check if the task has arrived
        if if_first_action_is_idle and if_second_action_is_idle:
            tmp_reward -= 50  # penalty for selecting two idle actions
        if not if_first_action_is_idle and not self.task_arrived[task1_id]:
            tmp_reward -= 100  # penalty for selecting a task that has not arrived
        if not if_second_action_is_idle and not self.task_arrived[task2_id]:
            tmp_reward -= 100  # penalty for selecting a task that has not arrived
        # check if select action for busy stream
        if not if_first_action_is_idle and self.stream_status[0]:
            tmp_reward -= 100  # penalty for selecting a task for busy stream
        if not if_second_action_is_idle and self.stream_status[1]:
            tmp_reward -= 100  # penalty for selecting a task for busy stream

        futures: List[concurrent.futures.Future] = []

        if not self.stream_status[0] and not if_first_action_is_idle:
            self.stream_status[0] = True
            future = self.executor.submit(
                self.execute_task,
                self.task_list[task1_id],
                task1_id,
                variant1_id,
                0,
                self.task_queues[task1_id][self.current_task_pointer[task1_id]][
                    "deadline"
                ],
            )
            self.futures.append(future)
            self.future_to_stream_index[future] = 0

        if not self.stream_status[1] and not if_second_action_is_idle:
            self.stream_status[1] = True
            future = self.executor.submit(
                self.execute_task,
                self.task_list[task2_id],
                task2_id,
                variant2_id,
                1,
                self.task_queues[task2_id][self.current_task_pointer[task2_id]][
                    "deadline"
                ],
            )
            self.futures.append(future)
            self.future_to_stream_index[future] = 1

        # Wait for any of the futures to complete
        done_futures, _ = concurrent.futures.wait(
            self.futures, return_when=concurrent.futures.FIRST_COMPLETED
        )

        for future in done_futures:
            result: Tuple[int, int] = future.result()
            stream_index: int = self.future_to_stream_index[future]
            self.stream_status[stream_index] = False
            self.futures.remove(future)
            # Update the reward based on the result
            tmp_reward += result[0] * 100 - result[1] * 100
            # remove the future from the future_to_stream_index
            del self.future_to_stream_index[future]

        current_time_ms: float = time.time() * 1000
        for task_id, queue in self.task_queues.items():
            if self.current_task_pointer[task_id] < len(queue):
                if_task_available: bool = False
                while (
                    self.current_task_pointer[task_id] < len(queue)
                    and queue[self.current_task_pointer[task_id]]["deadline"]
                    <= current_time_ms - self.start_time
                ):
                    self.current_task_pointer[task_id] += 1
                if (
                    self.current_task_pointer[task_id] < len(queue)
                    and queue[self.current_task_pointer[task_id]]["start_time"]
                    <= current_time_ms - self.start_time
                ):
                    self.task_arrived[task_id] = True
                    if_task_available = True
                if not if_task_available:
                    self.task_arrived[task_id] = False
            else:
                self.task_arrived[task_id] = False
        gpu_resources: Tuple[float, float] = get_gpu_resources()
        observation: Dict[str, Any] = {
            "current_streams_status": np.array(self.stream_status, dtype=np.float32),
            "current_time": np.array(
                [current_time_ms - self.start_time], dtype=np.float32
            ),
            "task_deadlines": np.array(
                [
                    (
                        queue[self.current_task_pointer[task_id]]["deadline"]
                        - (current_time_ms - self.start_time)
                        if self.current_task_pointer[task_id] < len(queue)
                        else -float("inf")
                    )
                    for task_id, queue in self.task_queues.items()
                ],
                dtype=np.float32,
            ),
            "task_if_arrived": np.array(self.task_arrived, dtype=np.float32),
            "task_if_periodic": np.array(self.if_periodic, dtype=np.float32),
            "variant_runtimes": self.variant_runtimes.astype(np.float32),
            "variant_accuracies": self.variant_accuracies.astype(np.float32),
            "gpu_resources": np.array(gpu_resources, dtype=np.float32),
        }
        reward: float = tmp_reward + 50 * (
            gpu_resources[0] + gpu_resources[1]
        )  # encourage to use GPU resources
        done: bool = current_time_ms - self.start_time >= self.total_time_ms
        info: Dict[str, Any] = {}

        return observation, reward, done, info

    def close(self) -> None:
        """
        Closes the environment and waits for all threads to complete.
        """
        self.executor.shutdown(wait=True)
        # clean up the GPU streams
        for stream in self.streams:
            stream.synchronize()
        del self.streams


if __name__ == "__main__":
    # Example usage:
    env: DLSchedulingEnv = DLSchedulingEnv(
        config_file="config.json", model_info_file="model_information.csv"
    )

    # Reset the environment to get the initial observation
    observation: Dict[str, Any] = env.reset()

    # Wrap the environment to use vectorized environments, which is required by stable-baselines3
    env = make_vec_env(lambda: env, n_envs=1)

    # Define a callback to evaluate and stop training once a reward threshold is achieved
    eval_callback = EvalCallback(
        env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=500,
        deterministic=True,
        render=False,
    )

    # Automatically select an available GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the PPO model
    model = PPO(
        "MultiInputPolicy", env, verbose=1, tensorboard_log="./logs/", device=device
    )

    # Train the model
    model.learn(total_timesteps=100000, callback=eval_callback)

    # Save the model
    model.save("ppo_dl_scheduling")

    # Load the model for further use or evaluation
    model = PPO.load("ppo_dl_scheduling")

    # Evaluate the model
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
