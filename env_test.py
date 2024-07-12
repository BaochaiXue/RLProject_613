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

if __name__ == "__main__":
    model = MaskablePPO.load("ppo_dl_scheduling")
    env = DLSchedulingEnv(
        config_file="config.json",
        model_info_file="model_information.csv",
        if_training=False,
    )
    env = ActionMasker(env, lambda env: env.valid_action_mask())
    env = make_vec_env(lambda: env, n_envs=1)
    obs = env.reset()
    predict_times: List[float] = []
    for i in range(10000):
        action_masks = env.env_method("valid_action_mask")
        action_masks = np.array(action_masks)
        action: np.ndarray
        start_predict = time.time()
        action, _states = model.predict(obs, action_masks=action_masks)
        stop_predict = time.time()
        predict_times.append(stop_predict - start_predict)
        obs, rewards, dones, info = env.step(action)

        if dones:
            env.render("human")
            obs = env.reset()
    print(
        f"Average prediction time: {np.mean(predict_times)}, std: {np.std(predict_times)} max: {np.max(predict_times)} min: {np.min(predict_times)}"
    )
    env.reset()
    step_times: List[float] = [0 for _ in range(env.task_num)]
    for i in range(10000):
        action_random_taken = i % env.task_num
        action = np.array([action_random_taken, 0, env.task_num, 0])
        start_step_time = time.time()
        obs, rewards, dones, info = env.step(action)
        stop_step_time = time.time()
        step_times[action_random_taken] += stop_step_time - start_step_time
        if dones:
            env.reset()
    for i in range(env.task_num):
        print(
            f"Average step time for task {i}: {step_times[i] / 10000}, std: {np.std(step_times[i])}"
        )
