import json
import gymnasium as gym
from gymnasium import spaces
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable, Tuple, Set
import time
import os
import random
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch import nn
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import MaskablePPO, TRPO
from sb3_contrib.common.wrappers import ActionMasker
import pynvml
import sys
from stable_baselines3 import PPO

pynvml.nvmlInit()

avg_predict_time: float = 1.22059
std_predict_time: float = 0.150174


def load_testset(vit_16_using: bool, dataset_name: str) -> Any:
    if dataset_name == "CIFAR10":
        transform = (
            transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                    ),
                ]
            )
            if vit_16_using
            else transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                    ),
                ]
            )
        )

        testset = datasets.CIFAR10(
            root="./data", train=False, transform=transform, download=True
        )
    elif dataset_name == "GTSRB":
        transform = (
            transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669]
                    ),
                ]
            )
            if not vit_16_using
            else transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669]
                    ),
                ]
            )
        )

        testset = datasets.GTSRB(
            root="./data", split="test", transform=transform, download=True
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return testset


testset_non_vit: datasets.CIFAR10 = load_testset(False, "CIFAR10")
testset_vit: datasets.CIFAR10 = load_testset(True, "CIFAR10")
testset_gtsrb: datasets.GTSRB = load_testset(False, "GTSRB")


def load_single_test_image(
    vit_16_using: bool, dataset_name: str = "CIFAR10"
) -> DataLoader:
    if dataset_name == "CIFAR10":
        testset: datasets.CIFAR10 = testset_vit if vit_16_using else testset_non_vit
    elif dataset_name == "GTSRB":
        testset: datasets.GTSRB = testset_gtsrb
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    random_index: int = random.randint(0, len(testset) - 1)
    test_subset: Subset = Subset(testset, [random_index])
    return DataLoader(test_subset, batch_size=1, shuffle=False)


def load_model(model_name: str, model_number: int, dataset_name: str) -> nn.Module:
    model_file: str = (
        f"selected_models/{model_name}/{model_name}_{dataset_name}_{model_number}.pth"
    )
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file {model_file} not found.")
    model: nn.Module = torch.load(model_file)
    model.eval()
    return model


def get_gpu_resources() -> Tuple[float, float]:
    device_count = pynvml.nvmlDeviceGetCount()
    total_gpu_util = 0
    total_mem_util = 0

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_gpu_util += util.gpu
        total_mem_util += mem_info.used / mem_info.total * 100

    avg_gpu_util = total_gpu_util / device_count if device_count > 0 else 0.0
    avg_mem_util = total_mem_util / device_count if device_count > 0 else 0.0

    return avg_gpu_util / 100.0, avg_mem_util / 100.0


class NGPSchedulingEnv(gym.Env):
    def __init__(
        self,
        config_file: str,
        model_info_file: str,
        if_training: bool = False,
        test_name: str = "MPPO",
    ) -> None:
        super(NGPSchedulingEnv, self).__init__()
        self.test_round_cnt: int = 0
        self.test_name: str = test_name
        self.if_training: bool = if_training
        self.train_time_record: float = 0.0

        with open(config_file, "r") as file:
            self.config: Dict[str, Any] = json.load(file)

        self.model_name_set: Set[str] = set()
        self.model_info: pd.DataFrame = pd.read_csv(model_info_file)

        self.initialize_parameters()
        self.define_spaces()

        self.total_tasks_count: List[int] = [0] * self.num_tasks
        self.total_task_actual_inference: List[int] = [0] * self.num_tasks
        self.total_task_accurate: List[int] = [0] * self.num_tasks
        self.total_missed_deadlines: List[int] = [0] * self.num_tasks
        self.task_queues: Dict[int, List[Dict[str, Any]]] = self.generate_task_queues()
        self.current_task_pointer: Dict[int, int] = {
            task_id: 0 for task_id in range(self.num_tasks)
        }
        self.if_periodic: List[bool] = [
            task.get("if_periodic", False) for task in self.task_list
        ]
        self.task_arrived: List[bool] = [False] * self.num_tasks

        self.models: Dict[Tuple[str, int], nn.Module] = {}
        for model_name in self.model_name_set:
            for variant_id in range(self.num_variants):
                self.models[(model_name, variant_id)] = load_model(
                    model_name,
                    variant_id + 1,
                    "cifar10" if "alexnet" not in model_name else "GTSRB",
                )
                self.models[(model_name, variant_id)].to("cuda:0")

        self.start_time: float = time.time() * 1000
        self._models_self_test()
        self._step_self_test()
        self._logs: pd.DataFrame = pd.DataFrame()
        self.start_time = time.time() * 1000

    def _models_self_test(self) -> None:
        if_training: bool = self.if_training
        self.if_training = True
        for task_id in range(self.num_tasks):
            for var_id in range(self.num_variants):
                self.execute_task(
                    self.task_list[task_id],
                    task_id,
                    var_id,
                    self.task_queues[task_id][0]["deadline"],
                )
            self.total_task_actual_inference[task_id] = 0
            self.total_task_accurate[task_id] = 0
            self.total_missed_deadlines[task_id] = 0
        self.if_training = if_training
        self.reset()

    def _step_self_test(self) -> None:
        if_training: bool = self.if_training
        self.if_training = True
        for task_id1 in range(self.num_tasks + 1):
            for var_id1 in range(self.num_variants):
                action = np.array([task_id1, var_id1])
                obs, reward, done, tun, info = self.step(action)
                if done:
                    self.reset()
        self.if_training = if_training
        self.reset()

    def initialize_parameters(self) -> None:
        self.num_tasks: int = self.config["num_tasks"]
        self.num_variants: int = self.config["num_variants"]
        self.total_time_ms: int = self.config["total_time_ms"]
        self.task_list: List[Dict[str, Any]] = self.config["task_list"]

        highest_periodic: int = max(
            [
                task["period_ms"]
                for task in self.task_list
                if task.get("if_periodic", False)
            ],
            default=0,
        )
        highest_deadline: int = max(
            [
                task["deadline_ms"]
                for task in self.task_list
                if not task.get("if_periodic")
            ],
            default=0,
        )
        self.max_deadline: int = max(highest_periodic, highest_deadline)
        self.variant_runtimes, self.variant_accuracies = self._extract_variant_info()

    def generate_task_queues(self) -> Dict[int, List[Dict[str, Any]]]:
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
                possion_lambda: int = task["possion_lambda"]
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
            self.total_tasks_count[task_id] = len(task_queue)
        return task_queues

    def _extract_variant_info(self) -> Tuple[np.ndarray, np.ndarray]:
        runtime_dict: Dict[str, List[float]] = {}
        accuracy_dict: Dict[str, List[float]] = {}
        column_max: float = self.model_info["Inference Time (s)"].max()

        for _, row in self.model_info.iterrows():
            model_name: str = row["Model Name"]
            self.model_name_set.add(model_name)
            variant_id: int = int(row["Model Number"]) - 1
            runtime: float = row["Inference Time (s)"] / column_max
            accuracy: float = row["Accuracy (Percentage)"] / 100

            if model_name not in runtime_dict:
                runtime_dict[model_name] = [None] * self.num_variants
                accuracy_dict[model_name] = [None] * self.num_variants

            runtime_dict[model_name][variant_id] = runtime
            accuracy_dict[model_name][variant_id] = accuracy

        runtimes = [runtime_dict[task["model"]] for task in self.task_list]
        accuracies = [accuracy_dict[task["model"]] for task in self.task_list]

        return np.array(runtimes), np.array(accuracies)

    def define_spaces(self) -> None:
        self.action_space = spaces.MultiDiscrete(
            [self.num_tasks + 1, self.num_variants]
        )

        self.observation_space = spaces.Dict(
            {
                "task_deadlines": spaces.Box(
                    low=0,
                    high=self.total_time_ms,
                    shape=(self.num_tasks,),
                    dtype=np.float32,
                ),
                "gpu_resources": spaces.Box(
                    low=0, high=1, shape=(2,), dtype=np.float32
                ),
            }
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.current_task_pointer = {task_id: 0 for task_id in range(self.num_tasks)}
        self.total_tasks_count = [0] * self.num_tasks
        self.total_task_actual_inference = [0] * self.num_tasks
        self.total_task_accurate = [0] * self.num_tasks
        self.total_missed_deadlines = [0] * self.num_tasks
        self.task_arrived = [False] * self.num_tasks
        self.task_queues = self.generate_task_queues()
        self.start_time = time.time() * 1000
        task_if_arrived = [False] * self.num_tasks
        current_time_ms = time.time() * 1000 + 1e-6
        for task_id, queue in self.task_queues.items():
            if self.current_task_pointer[task_id] < len(queue):
                if_task_available = False
                while (
                    self.current_task_pointer[task_id] < len(queue)
                    and queue[self.current_task_pointer[task_id]]["deadline"]
                    <= current_time_ms - self.start_time
                ):
                    self.total_missed_deadlines[task_id] += 1
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
        task_if_arrived = np.array(task_if_arrived)
        self.task_arrived = task_if_arrived
        task_ddls = []
        for task_id, queue in self.task_queues.items():
            if (
                self.current_task_pointer[task_id] < len(queue)
                and self.task_arrived[task_id]
            ):
                task_ddls.append(
                    queue[self.current_task_pointer[task_id]]["deadline"]
                    - (current_time_ms - self.start_time)
                )
            else:
                task_ddls.append(self.max_deadline * 10)
        initial_observation = {
            "task_deadlines": np.array(task_ddls, dtype=np.float32),
            "gpu_resources": np.array(get_gpu_resources(), dtype=np.float32),
        }
        info = {}
        self.start_time = time.time() * 1000
        if self.if_training:
            self.train_time_record = time.time() * 1000
        return initial_observation, info

    def execute_task(
        self, task: Dict[str, Any], task_id: int, variant_id: int, deadline: float
    ) -> Tuple[int, int]:
        model: nn.Module = self.models[(task["model"], variant_id)]
        dataloader: DataLoader = load_single_test_image(
            "vit" in task["model"], "CIFAR10"
        )
        device: torch.device = torch.device("cuda:0")
        penalty_function: Callable[[float], float] = lambda x: (
            20 + x * 0.1 if x > 0 else -3 + x * 0.1
        )

        model.eval()
        model.to(device)
        correct: int = 0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs: torch.Tensor = model(images)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        finish_time: float = time.time() * 1000
        self.total_task_actual_inference[task_id] += 1
        if correct == 1:
            self.total_task_accurate[task_id] += 1
        if finish_time - self.start_time > deadline:
            self.total_missed_deadlines[task_id] += 1
        return correct * 0.1, penalty_function(
            (finish_time - self.start_time - deadline) / self.max_deadline
        )

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.if_training:
            current_time: float = time.time() * 1000
            if (
                current_time - self.train_time_record
                > avg_predict_time + std_predict_time * 2
                and not np.isclose(self.train_time_record, 0.0, atol=1e-6)
            ):
                self.start_time += current_time - self.train_time_record
                self.start_time -= np.abs(
                    np.random.normal(avg_predict_time, std_predict_time)
                )
        task_id: int
        variant_id: int
        task_id, variant_id = action
        variant_id %= self.num_variants
        if_first_action_is_idle: bool = task_id == self.num_tasks
        reward: float = 0.0
        if not if_first_action_is_idle and not self.task_arrived[task_id]:
            reward -= 1
        if not if_first_action_is_idle and self.current_task_pointer[task_id] >= len(
            self.task_queues[task_id]
        ):
            reward -= 1
        if (
            not if_first_action_is_idle
            and self.task_arrived[task_id]
            and self.current_task_pointer[task_id] < len(self.task_queues[task_id])
        ):
            self.task_arrived[task_id] = False
            result: Tuple[int, float] = self.execute_task(
                self.task_list[task_id],
                task_id,
                variant_id,
                self.task_queues[task_id][self.current_task_pointer[task_id]][
                    "deadline"
                ],
            )
            self.current_task_pointer[task_id] += 1
            reward += result[0] - result[1]

        current_time_ms: float = time.time() * 1000
        for task_id, queue in self.task_queues.items():
            if self.current_task_pointer[task_id] < len(queue):
                if_task_available: bool = False
                while (
                    self.current_task_pointer[task_id] < len(queue)
                    and queue[self.current_task_pointer[task_id]]["deadline"]
                    <= current_time_ms - self.start_time
                ):
                    reward -= 20
                    self.total_missed_deadlines[task_id] += 1
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
        task_ddls: List[float] = []
        for task_id, queue in self.task_queues.items():
            if (
                self.current_task_pointer[task_id] < len(queue)
                and self.task_arrived[task_id]
            ):
                task_ddls.append(
                    queue[self.current_task_pointer[task_id]]["deadline"]
                    - (current_time_ms - self.start_time)
                )
            else:
                task_ddls.append(self.max_deadline * 10)
        observation: Dict[str, Any] = {
            "task_deadlines": np.array(task_ddls, dtype=np.float32),
            "gpu_resources": np.array(gpu_resources, dtype=np.float32),
        }
        done: bool = all(
            [
                self.current_task_pointer[task_id] >= len(queue)
                and not self.task_arrived[task_id]
                for task_id, queue in self.task_queues.items()
            ]
        )
        info: Dict[str, Any] = {}
        if self.if_training:
            self.train_time_record = time.time() * 1000
        if not self.if_training and done:
            self.render("human")
        return observation, reward, done, done, info

    def close(self) -> None:
        if self.if_training:
            return
        self._logs.to_csv(f"my_log/{self.test_name}_logs.csv", index=False)

    def valid_action_mask(self) -> np.ndarray:
        task_mask1 = np.array([True] * self.num_tasks + [False], dtype=np.bool_)
        variant_mask1 = np.array([True] * self.num_variants, dtype=np.bool_)

        for i in range(self.num_tasks):
            if not self.task_arrived[i] or self.current_task_pointer[i] >= len(
                self.task_queues[i]
            ):
                task_mask1[i] = False

        if not np.any(task_mask1[: self.num_tasks]):
            task_mask1[self.num_tasks] = True
            variant_mask1[:] = False
            variant_mask1[0] = True
        else:
            task_mask1[self.num_tasks] = False

        action_mask = np.array([task_mask1, variant_mask1], dtype=np.bool_)
        return action_mask

    def render(self, mode: str = "human") -> None:
        total_task_finished: int = np.sum(self.total_tasks_count)
        total_task_accurate: int = np.sum(self.total_task_accurate)
        total_missed_deadlines: int = np.sum(self.total_missed_deadlines)
        ddl_miss_rate: float = total_missed_deadlines / total_task_finished
        accuracy_for_total: float = total_task_accurate / total_task_finished
        accuracy_for_actual_inference: float = total_task_accurate / np.sum(
            self.total_task_actual_inference
        )
        print(f"DDL Miss Rate: {ddl_miss_rate}")
        print(f"Accuracy of Total: {accuracy_for_total}")
        print(f"Accuracy of Actual Inference: {accuracy_for_actual_inference}")
        for task_id in range(self.num_tasks):
            print(
                f"task {task_id} missed deadlines: {self.total_missed_deadlines[task_id]}"
            )
        for task_id in range(self.num_tasks):
            row: Dict[str, Any] = {
                "test_count": self.test_round_cnt,
                "task_id": task_id,
                "total_task_count": self.total_tasks_count[task_id],
                "total_task_accurate": self.total_task_accurate[task_id],
                "total_missed_deadlines": self.total_missed_deadlines[task_id],
                "total_task_actual_inference": self.total_task_actual_inference[
                    task_id
                ],
            }
            self._logs = pd.concat([self._logs, pd.DataFrame([row])], ignore_index=True)
        self.test_round_cnt += 1


def train_TRPO(time_step_of_training: int, test_kind: str) -> None:
    workloads: List[str] = ["lw", "mw", "hw"]
    for workload in workloads:
        env: NGPSchedulingEnv = NGPSchedulingEnv(
            config_file=f"config_{workload}.json",
            model_info_file="model_information.csv",
            if_training=True,
            test_name=f"TRPO_{test_kind}_{workload}",
        )
        env = make_vec_env(lambda: env, n_envs=1)

        device: torch.device = torch.device("cpu")
        model: TRPO = TRPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log="./logs/",
            device=device,
            learning_rate=0.0003,
        )

        env.reset()
        model.learn(total_timesteps=time_step_of_training, progress_bar=True)
        model.save(f"gp_ngp_models/trpo_{test_kind}_{workload}")
        model = TRPO.load(f"gp_ngp_models/trpo_{test_kind}_{workload}")
        env.close()
        env = NGPSchedulingEnv(
            config_file=f"config_{workload}.json",
            model_info_file="model_information.csv",
            if_training=False,
            test_name=f"TRPO_{test_kind}_{workload}",
        )
        env = ActionMasker(env, lambda env: env.valid_action_mask())
        env = make_vec_env(lambda: env, n_envs=1)
        obs = env.reset()
        for _ in range(10000):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)

            if dones:
                obs = env.reset()
                break
        env.close()


def train_PPO(time_step_of_training: int, test_kind: str) -> None:
    workloads: List[str] = ["lw", "mw", "hw"]
    for workload in workloads:
        env: NGPSchedulingEnv = NGPSchedulingEnv(
            config_file=f"config_{workload}.json",
            model_info_file="model_information.csv",
            if_training=True,
            test_name=f"PPO_{test_kind}_{workload}",
        )
        env = make_vec_env(lambda: env, n_envs=1)

        device: torch.device = torch.device("cpu")
        model: PPO = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log="./logs/",
            device=device,
            learning_rate=0.0003,
        )

        env.reset()
        model.learn(total_timesteps=time_step_of_training, progress_bar=True)
        model.save(f"gp_ngp_models/ppo_{test_kind}_{workload}")
        model = PPO.load(f"gp_ngp_models/ppo_{test_kind}_{workload}")
        env.close()
        env = NGPSchedulingEnv(
            config_file=f"config_{workload}.json",
            model_info_file="model_information.csv",
            if_training=False,
            test_name=f"PPO_{test_kind}_{workload}",
        )
        env = ActionMasker(env, lambda env: env.valid_action_mask())
        env = make_vec_env(lambda: env, n_envs=1)
        obs = env.reset()
        for _ in range(10000):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)

            if dones:
                obs = env.reset()
                break
        env.close()


if __name__ == "__main__":
    test_kind: str = "ngp"
    if not os.path.exists("gp_ngp_models"):
        os.makedirs("gp_ngp_models")
    if len(sys.argv) != 2:
        print("Usage: environment.py <param1> ")
        raise ValueError("Invalid number of arguments")
    time_step_of_training: int = int(sys.argv[1])
    train_TRPO(time_step_of_training, test_kind)
    train_PPO(time_step_of_training, test_kind)