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


def test_IM():
    # IM means inference time monotonic scheduling
    pass


if __name__ == "__main__":
    print("Testing MPPO")
    test_MPPO()
    print("Testing DM")
    test_DM()
