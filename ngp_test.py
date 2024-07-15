import json
import gymnasium as gym
from gymnasium import spaces
import torch
import concurrent.futures
import pandas as pd
import numpy as np
import typing
from typing import Dict, List, Any, Callable, Tuple, Set
import time
import GPUtil
import os
import random
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch import nn
import threading
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
import pynvml
from stable_baselines3.common.callbacks import BaseCallback
from environment import DLSchedulingEnv
from sb3_contrib.common.maskable.utils import get_action_masks
import warnings
from env_detail import DetailedDLSchedulingEnv
from other_rl_trains import FlattenDLSchedulingEnv
from stable_baselines3 import DQN
from sb3_contrib import ARS
from sb3_contrib import TRPO
from stable_baselines3 import PPO
import warnings
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import QRDQN

warnings.filterwarnings("ignore")
inference_times_of_reinforcement_learning: Dict[str, List[float]] = {}


def test_TRPO(test_steps: int = 10000):
    model = TRPO.load("RL_models/trpo_dl_scheduling")
    env = DLSchedulingEnv(
        config_file="config.json",
        model_info_file="model_information.csv",
        if_training=False,
        test_name="TRPO",
    )
    env = ActionMasker(env, lambda env: env.valid_action_mask())
    env = make_vec_env(lambda: env, n_envs=1)
    inference_times_of_reinforcement_learning["TRPO"] = []
    obs = env.reset()
    have_reset: bool = False
    for i in range(test_steps):
        start_time_of_inference = time.time()
        action: np.ndarray
        action, _states = model.predict(obs, deterministic=True)
        end_time_of_inference = time.time()
        obs, rewards, dones, info = env.step(action)
        inference_times_of_reinforcement_learning["TRPO"].append(
            end_time_of_inference - start_time_of_inference
        )
        if dones or (i == test_steps - 1 and not have_reset):
            obs = env.reset()
            have_reset = True
    env.close()


def test_PPO(test_steps: int = 10000):
    model = PPO.load("RL_models/ppo_dl_scheduling")
    env = DLSchedulingEnv(
        config_file="config.json",
        model_info_file="model_information.csv",
        if_training=False,
        test_name="PPO",
    )
    env = ActionMasker(env, lambda env: env.valid_action_mask())
    env = make_vec_env(lambda: env, n_envs=1)
    inference_times_of_reinforcement_learning["PPO"] = []
    obs = env.reset()
    for i in range(test_steps):
        start_time_of_inference = time.time()
        action: np.ndarray
        action, _states = model.predict(obs, deterministic=True)
        end_time_of_inference = time.time()
        obs, rewards, dones, info = env.step(action)
        inference_times_of_reinforcement_learning["PPO"].append(
            end_time_of_inference - start_time_of_inference
        )
        if dones:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    test_list: List[str] = [
        "TRPO",
        "PPO",
    ]
    test_kinds: List[str] = ["gp", "ngp"]
    worloads: List[str] = ["lw", "mw", "hw"]
    test_steps = 20000

    for test_name in test_list:
        print(f"Testing {test_name}")
        if test_name == "TRPO":
            test_TRPO(test_steps)
        elif test_name == "PPO":
            test_PPO(test_steps)
        else:
            print(f"Invalid test name {test_name}")

    today_date: str = time.strftime("%Y-%m-%d")
    time_now: str = time.strftime("%H-%M-%S")  # Hyphens are fine here for filenames
    result_file = f"test_result/test_results_{today_date}_{time_now}.csv"
    result_df: pd.DataFrame = pd.DataFrame()
    for test_name in test_list:
        log_file = f"my_log/{test_name}_logs.csv"
        logs: pd.DataFrame = pd.read_csv(log_file)
        df_row: Dict[str, Any] = {
            "test_name": test_name,
            "total_task_count": logs["total_task_count"].sum(),
            "total_task_accurate": logs["total_task_accurate"].sum(),
            "total_missed_deadlines": logs["total_missed_deadlines"].sum(),
            "total_task_actual_inference": logs["total_task_actual_inference"].sum(),
        }
        total_task_count: int = logs["total_task_count"].sum()
        total_task_accurate: int = logs["total_task_accurate"].sum()
        total_missed_deadlines: int = logs["total_missed_deadlines"].sum()
        total_task_actual_inference: int = logs["total_task_actual_inference"].sum()
        accuracy: float = total_task_accurate / total_task_actual_inference
        missed_deadlines: float = total_missed_deadlines / total_task_count
        print(
            "====================================="
            + test_name
            + "====================================="
        )
        print(
            f"Test {test_name} accuracy: {accuracy}, missed deadlines: {missed_deadlines}"
        )
        df_row["accuracy"] = accuracy
        df_row["missed_deadlines"] = missed_deadlines
        task_id_count: int = logs["task_id"].nunique()
        for task_id in range(task_id_count):
            task_logs: pd.DataFrame = logs[logs["task_id"] == task_id]
            total_task_count: int = task_logs["total_task_count"].sum()
            total_task_accurate: int = task_logs["total_task_accurate"].sum()
            total_missed_deadlines: int = task_logs["total_missed_deadlines"].sum()
            total_task_actual_inference: int = task_logs[
                "total_task_actual_inference"
            ].sum()
            accuracy: float = total_task_accurate / total_task_actual_inference
            missed_deadlines: float = total_missed_deadlines / total_task_count
            print(
                f"Task {task_id} accuracy: {accuracy}, missed deadlines: {missed_deadlines}"
            )
            df_row[f"task_{task_id}_accuracy"] = accuracy
            df_row[f"task_{task_id}_missed_deadlines"] = missed_deadlines
        inference_times: List[float] = inference_times_of_reinforcement_learning[
            test_name
        ]
        inference_times: np.ndarray = np.array(inference_times)
        mean_inference_time: float = np.mean(inference_times)
        std_inference_time: float = np.std(inference_times)
        print(
            f"Test {test_name} mean inference time: {mean_inference_time}, std inference time: {std_inference_time}"
        )
        df_row["mean_inference_time"] = mean_inference_time
        df_row["std_inference_time"] = std_inference_time
        print()
        result_df = pd.concat([result_df, pd.DataFrame([df_row])], ignore_index=True)
    result_df.to_csv(result_file, index=False)
