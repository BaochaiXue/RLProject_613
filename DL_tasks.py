import os
from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import sys
from torchvision.models import VisionTransformer, resnet50, vgg16, mobilenet_v3_large
import json
import random
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Callable
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


def load_model(model_name: str, model_number: int) -> nn.Module:
    model_file = f"selected_models/{model_name}/{model_name}_{model_number}.pth"
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file {model_file} not found.")
    model = torch.load(model_file)
    return model


def check_inference_result(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> int:
    model.eval()
    correct: int = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images: torch.Tensor
            labels: torch.Tensor
            images, labels = images.to(device), labels.to(device)
            outputs: torch.Tensor = model(images)
            predicted: torch.Tensor
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    return 0 if correct == 1 else 1  # Return 0 if correct, 1 otherwise


def main() -> None:
    args: List[str] = sys.argv
    if len(args) != 3:
        raise ValueError("Please provide the model name and model number.")

    model_name: str = args[1]
    model_number: int = int(args[2])
    testloader: DataLoader = load_single_test_image(vit_16_using="vit" in model_name)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: nn.Module = load_model(model_name, model_number)
    model.to(device)
    result: int = check_inference_result(model, testloader, device)
    sys.exit(result)


if __name__ == "__main__":
    main()
