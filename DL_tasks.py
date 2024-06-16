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


def load_single_test_image(vit_16_using: bool) -> DataLoader:
    # Define the transformations based on whether ViT-16 is used
    if vit_16_using:
        transform: Callable[[Any], Any] = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                ),
            ]
        )
    else:
        transform: Callable[[Any], Any] = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                ),
            ]
        )

    # Load the CIFAR-10 test set
    testset: datasets.CIFAR10 = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # Randomly select one image from the test set
    random_index: int = random.randint(0, len(testset) - 1)
    test_subset: Subset[datasets.CIFAR10] = Subset(testset, [random_index])
    testloader: DataLoader = DataLoader(test_subset, batch_size=1, shuffle=False)

    return testloader


def main() -> None:
    args: List[str] = sys.argv
    if len(args) != 4:
        raise ValueError(
            "Please provide the model name, pruning factor, epochs, and iterations."
        )
    model_name: str = args[1]
    model_number: int = int(args[2])
    testloader: DataLoader = load_single_test_image(vit_16_using="vit" in model_name)
    candidateModel.evaluate(testloader, device)
    print(candidateModel)
    candidateModel.save_info_to_json()


if __name__ == "__main__":
    main()
