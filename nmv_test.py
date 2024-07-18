import json
import gymnasium as gym
from gymnasium import spaces
import torch
import concurrent.futures
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import time
import os
import random
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch import nn
import threading
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import MaskablePPO, TRPO
from sb3_contrib.common.wrappers import ActionMasker
import pynvml
from environment import DLSchedulingEnv
from stable_baselines3 import PPO
import warnings
from env_ngp import NGPSchedulingEnv
from env_nmv import NMVDLSchedulingEnv

warnings.filterwarnings("ignore")
pynvml.nvmlInit()

# Dictionary to store inference times for different models
inference_times_of_reinforcement_learning: Dict[str, List[float]] = {}


def test_model(model_name: str, test_kind: str, workload: str, test_steps: int = 10000):
    model = None
    model_path = f"gp_ngp_models/{model_name.lower()}_{test_kind}_{workload}"
    if model_name == "TRPO":
        model = TRPO.load(model_path)
    elif model_name == "PPO":
        model = PPO.load(model_path)
    else:
        print(f"Unsupported model name: {model_name}")
        return

    env = (
        DLSchedulingEnv(
            config_file=f"config_{workload}.json",
            model_info_file="model_information.csv",
            if_training=False,
            test_name=f"{model_name}_{test_kind}_{workload}",
        )
        if test_kind == "gp"
        else NMVDLSchedulingEnv(
            config_file=f"config_{workload}.json",
            model_info_file="model_information.csv",
            if_training=False,
            test_name=f"{model_name}_{test_kind}_{workload}",
        )
    )
    env = ActionMasker(env, lambda env: env.valid_action_mask())
    env = make_vec_env(lambda: env, n_envs=1)
    inference_times_of_reinforcement_learning[
        f"{model_name}_{test_kind}_{workload}"
    ] = []

    obs = env.reset()
    count_episodes: int = 0
    while count_episodes <= test_steps // 5000:
        start_time_of_inference = time.time()
        action, _states = model.predict(obs, deterministic=True)
        end_time_of_inference = time.time()

        obs, rewards, dones, info = env.step(action)
        inference_times_of_reinforcement_learning[
            f"{model_name}_{test_kind}_{workload}"
        ].append(end_time_of_inference - start_time_of_inference)

        if dones:
            obs = env.reset()
            count_episodes += 1

    env.close()


def log_results(
    test_list: List[str], test_kinds: List[str], workloads: List[str], result_file: str
):
    result_df = pd.DataFrame()
    for test_name in test_list:
        for test_kind in test_kinds:
            for workload in workloads:
                log_file = f"my_log/{test_name}_{test_kind}_{workload}_logs.csv"
                logs = pd.read_csv(log_file)
                df_row = {
                    "test_name": test_name,
                    "test_kind": test_kind,
                    "workload": workload,
                    "total_task_count": logs["total_task_count"].sum(),
                    "total_task_accurate": logs["total_task_accurate"].sum(),
                    "total_missed_deadlines": logs["total_missed_deadlines"].sum(),
                    "total_task_actual_inference": logs[
                        "total_task_actual_inference"
                    ].sum(),
                }

                total_task_count = logs["total_task_count"].sum()
                total_task_accurate = logs["total_task_accurate"].sum()
                total_missed_deadlines = logs["total_missed_deadlines"].sum()
                total_task_actual_inference = logs["total_task_actual_inference"].sum()

                accuracy = total_task_accurate / total_task_actual_inference
                missed_deadlines = total_missed_deadlines / total_task_count

                print(
                    f"====================================={test_name}_{test_kind}_{workload}====================================="
                )
                print(
                    f"Test {test_name}_{test_kind}_{workload} accuracy: {accuracy}, missed deadlines: {missed_deadlines}"
                )

                df_row["accuracy"] = accuracy
                df_row["missed_deadlines"] = missed_deadlines

                task_id_count = logs["task_id"].nunique()
                for task_id in range(task_id_count):
                    task_logs = logs[logs["task_id"] == task_id]
                    total_task_count = task_logs["total_task_count"].sum()
                    total_task_accurate = task_logs["total_task_accurate"].sum()
                    total_missed_deadlines = task_logs["total_missed_deadlines"].sum()
                    total_task_actual_inference = task_logs[
                        "total_task_actual_inference"
                    ].sum()

                    accuracy = total_task_accurate / total_task_actual_inference
                    missed_deadlines = total_missed_deadlines / total_task_count

                    print(
                        f"Task {task_id} accuracy: {accuracy}, missed deadlines: {missed_deadlines}"
                    )

                    df_row[f"task_{task_id}_accuracy"] = accuracy
                    df_row[f"task_{task_id}_missed_deadlines"] = missed_deadlines

                inference_times = inference_times_of_reinforcement_learning[
                    f"{test_name}_{test_kind}_{workload}"
                ]
                mean_inference_time = np.mean(inference_times)
                std_inference_time = np.std(inference_times)

                print(
                    f"Test {test_name}_{test_kind}_{workload} mean inference time: {mean_inference_time}, std inference time: {std_inference_time}"
                )

                df_row["mean_inference_time"] = mean_inference_time
                df_row["std_inference_time"] = std_inference_time
                print()

                result_df = pd.concat(
                    [result_df, pd.DataFrame([df_row])], ignore_index=True
                )

    result_df.to_csv(result_file, index=False)


if __name__ == "__main__":
    test_list = ["TRPO", "PPO"]
    test_kinds = ["gp", "nmv"]
    workloads = ["lw", "mw", "hw"]
    test_steps = 30000

    for test_name in test_list:
        for test_kind in test_kinds:
            for workload in workloads:
                print(f"Testing {test_name} with {test_kind} workload {workload}")
                test_model(test_name, test_kind, workload, test_steps)

    today_date = time.strftime("%Y-%m-%d")
    time_now = time.strftime("%H-%M-%S")
    result_file = f"test_result/test_results_nmv_{today_date}_{time_now}.csv"

    log_results(test_list, test_kinds, workloads, result_file)
