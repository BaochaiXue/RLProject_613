import json
import gym
from gym import spaces
import torch
import threading
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

# Define the event object for stream availability
stream_available_event = threading.Event()


def load_single_test_image(vit_16_using: bool) -> DataLoader:
    if vit_16_using:
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                ),
            ]
        )

    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    random_index = random.randint(0, len(testset) - 1)
    test_subset = Subset(testset, [random_index])
    testloader = DataLoader(test_subset, batch_size=1, shuffle=False)
    return testloader


def load_model(model_name: str, model_number: int) -> nn.Module:
    model_file = f"selected_models/{model_name}/{model_name}_{model_number}.pt"
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file {model_file} not found.")
    model = torch.load(model_file)
    return model


class DLSchedulingEnv(gym.Env):
    def __init__(self, config_file: str, model_info_file: str):
        super(DLSchedulingEnv, self).__init__()

        with open(config_file, "r") as file:
            self.config = json.load(file)

        self.model_info = pd.read_csv(model_info_file)
        self.start_time = time.time() * 1000
        self.initialize_parameters()
        self.define_spaces()
        self.task_queues = self.generate_task_queues()
        self.current_task_pointer = {task_id: 0 for task_id in range(self.num_tasks)}
        self.if_periodic = [task.get("if_periodic", False) for task in self.task_list]
        self.streams = [torch.cuda.Stream(priority=-1), torch.cuda.Stream(priority=0)]
        self.stream_status = [False, False]  # False means idle, True means busy
        self.task_arrived = [False] * self.num_tasks
        self.task_threads = {}
        self.lock = threading.Lock()
        self.total_task_finished = [0] * self.num_tasks
        self.total_task_accurate = [0] * self.num_tasks
        self.total_missed_deadlines = [0] * self.num_tasks
        self.thread_results: Dict[int, typing.Tuple[int, int]] = (
            {}
        )  # stream_id -> (correct, missed_deadline)

    def get_gpu_resources() -> Tuple[float, float]:
        """
        Retrieves the GPU utility rate and GPU memory utility rate of all GPUs.

        Returns:
            Tuple[float, float]: A tuple containing the average GPU utility rate and
                                the average GPU memory utility rate of all GPUs.
        """
        # Retrieve a list of all GPUs
        gpus = GPUtil.getGPUs()

        # Initialize variables to store the sum of utility rates
        total_gpu_util = 0.0
        total_mem_util = 0.0

        # Iterate over all available GPUs
        gpu: GPUtil.GPU
        for gpu in gpus:
            total_gpu_util += gpu.load
            total_mem_util += gpu.memoryUtil

        # Calculate the average utility rates
        num_gpus = len(gpus)
        avg_gpu_util = total_gpu_util / num_gpus if num_gpus > 0 else 0.0
        avg_mem_util = total_mem_util / num_gpus if num_gpus > 0 else 0.0

        return avg_gpu_util, avg_mem_util

    def generate_task_queues(self) -> Dict[int, List[Dict[str, Any]]]:
        task_queues = {}
        for task_id, task in enumerate(self.task_list):
            task_queue = []
            current_time = 0
            if task.get("if_periodic", False):
                period_ms = task["period_ms"]
                while current_time < self.total_time_ms:
                    task_queue.append(
                        {
                            "start_time": current_time,
                            "deadline": current_time + period_ms,
                        }
                    )
                    current_time += period_ms
            else:
                possion_lambda = task["possion_lambda"]
                while current_time < self.total_time_ms:
                    inter_arrival_time = np.random.poisson(possion_lambda)
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
        self.num_tasks = self.config["num_tasks"]
        self.num_variants = self.config["num_variants"]
        self.total_time_ms = self.config["total_time_ms"]
        self.task_list = self.config["task_list"]

        self.variant_runtimes, self.variant_accuracies = self._extract_variant_info()

    def _extract_variant_info(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        runtime_dict = {}
        accuracy_dict = {}
        for _, row in self.model_info.iterrows():
            model_name = row["Model Name"]
            variant_id = int(row["Model Number"]) - 1
            runtime = row["Inference Time (s)"] * 1000
            accuracy = row["Accuracy (Percentage)"]
            if model_name not in runtime_dict:
                runtime_dict[model_name] = [None] * self.num_variants
                accuracy_dict[model_name] = [None] * self.num_variants
            runtime_dict[model_name][variant_id] = runtime
            accuracy_dict[model_name][variant_id] = accuracy

        runtimes = []
        accuracies = []
        for task in self.task_list:
            model_name = task["model"]
            runtimes.append(runtime_dict[model_name])
            accuracies.append(accuracy_dict[model_name])
        return np.array(runtimes), np.array(accuracies)

    def define_spaces(self) -> None:
        self.action_space = spaces.Dict(
            {
                "task1_id": spaces.Discrete(self.num_tasks + 1),
                "variant1_id": spaces.Discrete(self.num_variants),
                "task2_id": spaces.Discrete(self.num_tasks + 1),
                "variant2_id": spaces.Discrete(self.num_variants),
            }
        )

        self.observation_space = spaces.Dict(
            {
                "current_streams_status": spaces.MultiBinary(2),
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
        self.current_task_pointer = {task_id: 0 for task_id in range(self.num_tasks)}
        self.task_start_times = {}
        self.task_end_times = {}
        self.start_time = time.time() * 1000
        task_if_arrived = np.zeros(self.num_tasks)
        current_time_ms = time.time() * 1000
        for task_id, queue in self.task_queues.items():
            if self.current_task_pointer[task_id] < len(queue):
                if_task_available = False
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
                    task_if_arrived[task_id] = True
                    if_task_available = True
                if not if_task_available:
                    task_if_arrived[task_id] = False
            else:
                task_if_arrived[task_id] = False
        self.task_arrived = task_if_arrived
        initial_observation = {
            "current_streams_status": [False, False],
            "current_time": np.array([current_time_ms - self.start_time]),
            "task_deadlines": np.array(
                [
                    (
                        queue[self.current_task_pointer[task_id]]["deadline"]
                        - (current_time_ms - self.start_time)
                        if self.current_task_pointer[task_id] < len(queue)
                        else -float("inf")
                    )
                    for task_id, queue in self.task_queues.items()
                ]
            ),
            "task_if_arrived": task_if_arrived,
            "task_if_periodic": np.array(self.if_periodic),
            "variant_runtimes": self.variant_runtimes,
            "variant_accuracies": self.variant_accuracies,
            "gpu_resources": self.get_gpu_resources(),
        }

        return initial_observation

    def execute_task(
        self,
        task: Dict[str, Any],
        task_id: int,
        variant_id: int,
        stream_index: int,  # 0 or 1
        deadline: float,
        event: threading.Event,
    ) -> None:
        model = load_model(task["model"], variant_id + 1)
        dataloader = load_single_test_image("vit" in task["model"])
        device = torch.device(f"cuda:{stream_index}")
        stream = self.streams[stream_index]
        # update the pointer
        self.current_task_pointer[task_id] += 1
        self.task_arrived[task_id] = False

        def task_thread():
            nonlocal event
            self.stream_status[stream_index] = True
            with torch.cuda.stream(stream):
                model.eval()
                correct = 0
                model.to(device)
                with torch.no_grad():
                    for images, labels in dataloader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        correct += (predicted == labels).sum().item()
                # Update the total task finished and accuracy counts
                with self.lock:
                    self.total_task_finished[task_id] += 1
                    if correct == 1:
                        self.total_task_accurate[task_id] += 1
                    if time.time() * 1000 - self.start_time > deadline:
                        self.total_missed_deadlines[task_id] += 1
                    self.thread_results[task_id] = (
                        correct,
                        1 if time.time() * 1000 - self.start_time > deadline else 0,
                    )
            # Update the stream status
            self.stream_status[stream_index] = False
            event.set()

        thread = threading.Thread(target=task_thread)
        thread.start()
        self.task_threads[task_id] = thread

    def step(
        self, action: Dict[str, int]
    ) -> typing.Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        task1_id, variant1_id = action["task1_id"], action["variant1_id"]
        task2_id, variant2_id = action["task2_id"], action["variant2_id"]

        if_first_action_is_idle = task1_id == self.num_tasks
        if_second_action_is_idle = task2_id == self.num_tasks
        tmp_reward = 0.0
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
        if not self.stream_status[0] and not if_first_action_is_idle:
            self.execute_task(
                self.task_list[task1_id],
                task1_id,
                variant1_id,
                0,
                stream_available_event,
            )
        if not self.stream_status[1] and not if_second_action_is_idle:
            self.execute_task(
                self.task_list[task2_id],
                task2_id,
                variant2_id,
                1,
                stream_available_event,
            )
        while not any(not status for status in self.stream_status):
            stream_available_event.wait()
            stream_available_event.clear()
        current_time_ms = time.time() * 1000
        for task_id, queue in self.task_queues.items():
            if self.current_task_pointer[task_id] < len(queue):
                if_task_available = False
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
        gpu_resources = self.get_gpu_resources()
        observation = {
            "current_streams_status": self.stream_status,
            "current_time": np.array([current_time_ms - self.start_time]),
            "task_deadlines": np.array(
                [
                    (
                        queue[self.current_task_pointer[task_id]]["deadline"]
                        - (current_time_ms - self.start_time)
                        if self.current_task_pointer[task_id] < len(queue)
                        else -float("inf")
                    )
                    for task_id, queue in self.task_queues.items()
                ]
            ),
            "task_if_arrived": self.task_arrived,
            "task_if_periodic": np.array(self.if_periodic),
            "variant_runtimes": self.variant_runtimes,
            "variant_accuracies": self.variant_accuracies,
            "gpu_resources": gpu_resources,
        }
        reward = (
            tmp_reward
            + (self.thread_results[0][0] * 100 if self.stream_status[0] else 0)
            + (self.thread_results[1][0] * 100 if self.stream_status[1] else 0)
            - (self.thread_results[0][1] * 100 if self.stream_status[0] else 0)
            - (self.thread_results[1][1] * 100 if self.stream_status[1] else 0)
            + 50
            * (gpu_resources[0] + gpu_resources[1])  # encourage to use GPU resources
        )
        done = current_time_ms - self.start_time >= self.total_time_ms
        info = {}

        return observation, reward, done, info

    def close(self) -> None:
        for thread in self.task_threads.values():
            thread.join()
        del self.streams


# Example usage:
env = DLSchedulingEnv(
    config_file="config.json", model_info_file="model_information.csv"
)

# Reset environment to get the initial observation
observation = env.reset()
