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


class StopTrainingOnTimeLimit(BaseCallback):
    def __init__(self, time_limit_seconds: int, verbose=0):
        super(StopTrainingOnTimeLimit, self).__init__(verbose)
        self.time_limit_seconds = time_limit_seconds
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if time.time() - self.start_time >= self.time_limit_seconds:
            print("Time limit reached.")
            return False  # Returning False stops the training
        return True

    def _on_training_end(self) -> None:
        print("Training stopped due to time limit.")


if __name__ == "__main__":
    for _device in ["cpu", "cuda"]:
        env = DLSchedulingEnv(
            config_file="config.json",
            model_info_file="model_information.csv",
            if_training=False,
        )
        task_num: int = env.num_tasks
        env = ActionMasker(env, lambda env: env.valid_action_mask())
        env = make_vec_env(lambda: env, n_envs=1)

        device: torch.device = torch.device(device=_device)
        model: MaskablePPO = MaskablePPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log="./logs/",
            device=device,
            learning_rate=0.0003,
        )

        eval_callback: MaskableEvalCallback = MaskableEvalCallback(
            env,
            best_model_save_path="./logs/",
            log_path="./logs/",
            eval_freq=500,
            deterministic=True,
            render=False,
        )
        stopTrainingOnTimeLimit: StopTrainingOnTimeLimit = StopTrainingOnTimeLimit(10)
        env.reset()
        model.learn(
            total_timesteps=100000, callback=[stopTrainingOnTimeLimit, eval_callback]
        )
        model.save("ppo_dl_scheduling" + _device)
        obs = env.reset()
        predict_times: List[float] = []
        step_times: List[float] = []
        for i in range(10000):
            action_masks = env.env_method("valid_action_mask")
            action_masks = np.array(action_masks)
            action: np.ndarray
            start_predict = time.time()
            action, _states = model.predict(
                obs, action_masks=action_masks, deterministic=True
            )
            stop_predict = time.time()
            predict_times.append(stop_predict - start_predict)
            step_start = time.time()
            obs, rewards, dones, info = env.step(action)
            step_stop = time.time()
            step_times.append(step_stop - step_start)
            if dones:
                env.render("human")
                obs = env.reset()
        print(
            _device
            + f" Average prediction time: {np.mean(predict_times)}, std: {np.std(predict_times)} max: {np.max(predict_times)} min: {np.min(predict_times)}"
        )
        print(
            f"Average step time: {np.mean(step_times)}, std: {np.std(step_times)} max: {np.max(step_times)} min: {np.min(step_times)}"
        )
        env.close()
