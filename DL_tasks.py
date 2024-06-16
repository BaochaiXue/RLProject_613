import os
from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict
from torch.utils.data import DataLoader
import time
import sys
from torchvision.models import VisionTransformer
import json
import random
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Callable
from typing import Any
from typing import Tuple
from typing import List
from typing import Dict
from typing import Any


def load_data(batch_size: int, vit_16_using: bool) -> Tuple[DataLoader, DataLoader]:
    transform: Callable[[Any], Any] = (
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
    trainset: datasets.CIFAR10 = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader: DataLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset: datasets.CIFAR10 = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    random_index: int = random.randint(0, len(testset) - 1)
    test_subset: Subset[datasets.CIFAR10] = Subset(testset, [random_index])
    testloader: DataLoader = DataLoader(test_subset, batch_size=1, shuffle=False)

    return trainloader, testloader


class CandidateModel:

    def __init__(self, model_path: str, model_name: str) -> None:
        self.model: nn.Module = torch.load(model_path)
        self.size: int = os.path.getsize(model_path)
        self.name: str = model_name
        self.prune_rate: float = (
            0.0
            if "original" in model_path
            else float(model_path.split("_")[-1].split(".pt")[0])
        )
        self.accuracy: float = None
        self.inference_time: float = None

    def __repr__(self) -> str:
        return f"Model: {self.name}, Prune Rate: {self.prune_rate}, Accuracy: {self.accuracy:.2f}%, Inference Time: {self.inference_time:.4f} seconds, Model Size: {self.size} bytes"

    def __str__(self) -> str:
        return self.__repr__()

    def __lt__(self, other: "CandidateModel") -> bool:
        return self.size < other.size

    def __eq__(self, other: "CandidateModel") -> bool:
        return self.size == other.size

    def __gt__(self, other: "CandidateModel") -> bool:
        return self.size > other.size

    def __le__(self, other: "CandidateModel") -> bool:
        return self.size <= other.size

    def __ge__(self, other: "CandidateModel") -> bool:
        return self.size >= other.size

    def __ne__(self, other: "CandidateModel") -> bool:
        return self.size != other.size

    def evaluate(
        self, dataloader: DataLoader, device: torch.device
    ) -> Tuple[float, float]:
        self.model.to(device)
        self.model.eval()
        correct: int = 0
        total: int = 0
        start_time: float = time.time()
        with torch.no_grad():
            inputs: torch.Tensor
            targets: torch.Tensor
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs: torch.Tensor = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        end_time: float = time.time()
        inference_time: float = end_time - start_time
        accuracy: float = 100.0 * correct / total
        # clean up
        torch.cuda.empty_cache()
        self.accuracy = accuracy
        self.inference_time = inference_time
        return accuracy, inference_time

    def save_info_to_json(self, file_path: str = "model_info.json") -> None:
        model_info: Dict[str, Any] = {
            "model": self.name,
            "prune_rate": self.prune_rate,
            "accuracy": self.accuracy,
            "inference_time": self.inference_time,
            "model_size": self.size,
        }
        dir_name: str = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        # Save the model information to the JSON file
        with open(file_path, "w") as f:
            json.dump(model_info, f, indent=4)


def main() -> None:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args: List[str] = sys.argv
    if len(args) != 4:
        raise ValueError(
            "Please provide the model name, pruning factor, epochs, and iterations."
        )
    batch_size: int = int(args[1])
    path: str = args[2]
    model_name: str = args[3]
    candidateModel: CandidateModel = CandidateModel(path, model_name)
    testloader: DataLoader
    _, testloader = load_data(batch_size, "vit" in candidateModel.name)
    candidateModel.evaluate(testloader, device)
    print(candidateModel)
    candidateModel.save_info_to_json()


if __name__ == "__main__":
    main()
