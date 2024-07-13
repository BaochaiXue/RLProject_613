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

warnings.filterwarnings("ignore")


def test_MPPO():
    model = MaskablePPO.load("ppo_dl_scheduling")
    env = DLSchedulingEnv(
        config_file="config.json",
        model_info_file="model_information.csv",
        if_training=False,
    )
    env = ActionMasker(env, lambda env: env.valid_action_mask())
    env = make_vec_env(lambda: env, n_envs=1)
    obs = env.reset()
    for i in range(10000):
        action_masks = get_action_masks(env)
        action: np.ndarray
        action, _states = model.predict(
            obs, action_masks=action_masks, deterministic=True
        )
        start_time = time.time()
        obs, rewards, dones, info = env.step(action)
        end_time = time.time()
        # print(f"Time: {end_time - start_time}")
        if dones:
            obs = env.reset()
    env.close()


def test_DM():
    # DM means deadline monotonic scheduling
    env = DLSchedulingEnv(
        config_file="config.json",
        model_info_file="model_information.csv",
        if_training=False,
        test_name="DM",
    )
    obs: np.ndarray
    info: Dict[str, Any]
    obs, info = env.reset()
    for i in range(10000):
        action: np.ndarray
        num_tasks: int = env.num_tasks
        num_variants: int = env.num_variants
        # we check in the obs
        # maskable_action = env.valid_action_mask()
        current_stream_status_is_busy: np.ndarray = obs["current_streams_status"]
        task_deadlines: np.ndarray = obs["task_deadlines"]
        smallest_deadline_task_idx: int = np.argmin(task_deadlines)
        second_smallest_deadline_task_idx: int = np.argmin(
            np.where(task_deadlines == np.min(task_deadlines), np.inf, task_deadlines)
        )
        if current_stream_status_is_busy[0] and current_stream_status_is_busy[1]:
            action = np.array([num_tasks, 0, num_tasks, 0])
        elif current_stream_status_is_busy[0] and not current_stream_status_is_busy[1]:
            action = np.array([num_tasks, 0, smallest_deadline_task_idx, 0])
        elif not current_stream_status_is_busy[0] and current_stream_status_is_busy[1]:
            action = np.array([smallest_deadline_task_idx, 0, num_tasks, 0])
        else:
            action = np.array(
                [smallest_deadline_task_idx, 0, second_smallest_deadline_task_idx, 0]
            )
        obs, rewards, dones, tun, info = env.step(action)
        # time.sleep(0.001)
        if dones:
            obs, info = env.reset()
    env.close()


def test_DMI():
    # DM means deadline monotonic scheduling inverted
    env = DLSchedulingEnv(
        config_file="config.json",
        model_info_file="model_information.csv",
        if_training=False,
        test_name="DMI",
    )
    obs: np.ndarray
    info: Dict[str, Any]
    obs, info = env.reset()
    for i in range(10000):
        action: np.ndarray
        num_tasks: int = env.num_tasks
        num_variants: int = env.num_variants
        # we check in the obs
        # maskable_action = env.valid_action_mask()
        current_stream_status_is_busy: np.ndarray = obs["current_streams_status"]
        task_deadlines: np.ndarray = obs["task_deadlines"]
        biggest_deadline_task_idx: int = np.argmax(task_deadlines)
        second_biggest_deadline_task_idx: int = np.argmax(
            np.where(task_deadlines == np.max(task_deadlines), -np.inf, task_deadlines)
        )
        if current_stream_status_is_busy[0] and current_stream_status_is_busy[1]:
            action = np.array([num_tasks, 0, num_tasks, 0])
        elif current_stream_status_is_busy[0] and not current_stream_status_is_busy[1]:
            action = np.array([num_tasks, 0, biggest_deadline_task_idx, 0])
        elif not current_stream_status_is_busy[0] and current_stream_status_is_busy[1]:
            action = np.array([biggest_deadline_task_idx, 0, num_tasks, 0])
        else:
            action = np.array(
                [biggest_deadline_task_idx, 0, second_biggest_deadline_task_idx, 0]
            )
        obs, rewards, dones, tun, info = env.step(action)

        if dones or tun:
            obs, info = env.reset()
    env.close()


def test_STF():
    # IM shortest task first
    env = DetailedDLSchedulingEnv(
        config_file="config.json",
        model_info_file="model_information.csv",
        if_training=False,
        test_name="STF",
    )
    obs: np.ndarray
    info: Dict[str, Any]
    obs, info = env.reset()
    for i in range(10000):
        action: np.ndarray
        num_tasks: int = env.num_tasks
        num_variants: int = env.num_variants
        # we check in the obs
        # maskable_action = env.valid_action_mask()
        current_stream_status_is_busy: np.ndarray = obs["current_streams_status"]
        infer_time: np.ndarray = obs["infer_time"].copy()
        if_arrived: np.ndarray = obs["if_arrived"].copy()
        infer_time = [infer_time[task_id][0] for task_id in range(num_tasks)]
        if_arrived = [
            False if np.isclose(if_arrived[task_id], 0) else True
            for task_id in range(num_tasks)
        ]
        # if not arrived, we set the infer time to be inf
        infer_time = [
            np.inf if not if_arrived[task_id] else infer_time[task_id]
            for task_id in range(num_tasks)
        ]
        smallest_infer_time_task_idx: int = np.argmin(infer_time)
        second_smallest_infer_time_task_idx: int = np.argmin(
            np.where(infer_time == np.min(infer_time), np.inf, infer_time)
        )
        if current_stream_status_is_busy[0] and current_stream_status_is_busy[1]:
            action = np.array([num_tasks, 0, num_tasks, 0])
        elif current_stream_status_is_busy[0] and not current_stream_status_is_busy[1]:
            action = np.array([num_tasks, 0, smallest_infer_time_task_idx, 0])
        elif not current_stream_status_is_busy[0] and current_stream_status_is_busy[1]:
            action = np.array([smallest_infer_time_task_idx, 0, num_tasks, 0])
        else:
            action = np.array(
                [
                    smallest_infer_time_task_idx,
                    0,
                    second_smallest_infer_time_task_idx,
                    0,
                ]
            )
        obs, rewards, dones, tun, info = env.step(action)

        if dones or tun:
            obs, info = env.reset()
    env.close()


def test_FIFO():
    # FIFO first in first out
    env = DetailedDLSchedulingEnv(
        config_file="config.json",
        model_info_file="model_information.csv",
        if_training=False,
        test_name="FIFO",
    )
    obs: np.ndarray
    info: Dict[str, Any]
    obs, info = env.reset()

    for i in range(10000):
        action: np.ndarray
        num_tasks: int = env.num_tasks
        num_variants: int = env.num_variants
        # we check in the obs
        # maskable_action = env.valid_action_mask()
        current_stream_status_is_busy: np.ndarray = obs["current_streams_status"]
        task_arrival_time: np.ndarray = obs["start_time"]
        smallest_arrival_time_task_idx: int = np.argmin(task_arrival_time)
        second_smallest_arrival_time_task_idx: int = np.argmin(
            np.where(
                task_arrival_time == np.min(task_arrival_time),
                np.inf,
                task_arrival_time,
            )
        )
        if current_stream_status_is_busy[0] and current_stream_status_is_busy[1]:
            action = np.array([num_tasks, 0, num_tasks, 0])
        elif current_stream_status_is_busy[0] and not current_stream_status_is_busy[1]:
            action = np.array([num_tasks, 0, smallest_arrival_time_task_idx, 0])
        elif not current_stream_status_is_busy[0] and current_stream_status_is_busy[1]:
            action = np.array([smallest_arrival_time_task_idx, 0, num_tasks, 0])
        else:
            action = np.array(
                [
                    smallest_arrival_time_task_idx,
                    0,
                    second_smallest_arrival_time_task_idx,
                    0,
                ]
            )
        obs, rewards, dones, tun, info = env.step(action)

        if dones or tun:
            obs, info = env.reset()
    env.close()


def test_DQN():
    model = DQN.load("dqn_dl_scheduling")
    env = FlattenDLSchedulingEnv(
        config_file="config.json",
        model_info_file="model_information.csv",
        if_training=False,
    )
    env = make_vec_env(lambda: env, n_envs=1)
    obs = env.reset()
    for i in range(10000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        if dones:
            obs = env.reset()
            break
    env.close()


if __name__ == "__main__":
    test_list: List[str] = ["MPPO", "DQN"]
    print("Testing MPPO")
    test_MPPO()
    print("Testing DQN")
    test_DQN()
    for test_name in test_list:
        # read the log file
        log_file = f"{test_name}_logs.csv"
        logs: pd.DataFrame = pd.read_csv(log_file)
        # columns test_count,task_id,total_task_count,total_task_accurate,total_missed_deadlines,total_task_actual_inference
        # we can calculate the average accuracy
        total_task_count: int = logs["total_task_count"].sum()
        total_task_accurate: int = logs["total_task_accurate"].sum()
        total_missed_deadlines: int = logs["total_missed_deadlines"].sum()
        total_task_actual_inference: int = logs["total_task_actual_inference"].sum()
        accuracy: float = total_task_accurate / total_task_actual_inference
        missed_deadlines: float = total_missed_deadlines / total_task_count
        print(
            f"Test {test_name} accuracy: {accuracy}, missed deadlines: {missed_deadlines}"
        )
