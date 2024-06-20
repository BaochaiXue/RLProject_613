import os
import csv
import time
import torch
from typing import List, Tuple, Any
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
import typing
from typing import Callable


def get_inference_time_and_accuracy(
    model: torch.nn.Module, dataloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate the model's inference time and accuracy.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        dataloader (DataLoader): The dataloader for test data.
        device (torch.device): The device to run the model on.

    Returns:
        Tuple[float, float]: Average inference time and accuracy.
    """
    model.eval()
    correct: int = 0
    total: int = 0
    total_time: float = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images: torch.Tensor
            labels: torch.Tensor
            images, labels = images.to(device), labels.to(device)
            start_time: float = time.time()
            outputs: torch.Tensor = model(images)
            predicted: torch.Tensor
            _, predicted = torch.max(outputs.data, 1)
            total_time += time.time() - start_time
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    average_time: float = total_time / total
    accuracy: float = 100 * correct / total
    # clearing the cache
    torch.cuda.empty_cache()
    return average_time, accuracy


def rename_models_and_evaluate(
    directory: str,
    models_name: List[str],
    csv_file: str,
    device: torch.device,
) -> None:
    """
    Rename models and evaluate their performance.

    Args:
        directory (str): The directory containing the models.
        models_name (List[str]): The list of model names.
        csv_file (str): The CSV file to save the results.
        device (torch.device): The device to run the models on.
    """
    results: List[List[typing.Any]] = []

    for model_name in models_name:
        if "vit" in model_name:
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

        test_dataset: CIFAR10 = CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
        test_dataloader: DataLoader = DataLoader(
            test_dataset, batch_size=1, shuffle=False
        )

        model_directory: str = os.path.join(directory, model_name)
        try:
            if not os.path.exists(model_directory) or not os.path.isdir(
                model_directory
            ):
                raise FileNotFoundError(
                    f"Directory {model_directory} does not exist or is not a directory."
                )

            # Get all files in the model directory
            files: List[str] = [
                f
                for f in os.listdir(model_directory)
                if os.path.isfile(os.path.join(model_directory, f))
            ]

            if not files:
                raise FileNotFoundError(
                    f"No files found in directory {model_directory}."
                )

            # Get file sizes and sort by size from largest to smallest
            model_files_with_sizes: List[Tuple[str, int]] = [
                (f, os.path.getsize(os.path.join(model_directory, f))) for f in files
            ]
            model_files_with_sizes.sort(key=lambda x: x[1], reverse=True)

            # Load the model, measure inference time and accuracy, and rename files
            for idx, (file, size) in enumerate(model_files_with_sizes, start=1):
                old_path: str = os.path.join(model_directory, file)
                file_extension: str = os.path.splitext(file)[1]
                new_name: str = f"{model_name}_{idx}{file_extension}"
                new_path: str = os.path.join(model_directory, new_name)

                # Load the model
                model: torch.nn.Module = torch.load(old_path)
                model.to(device)

                # Get inference time and accuracy
                avg_time, accuracy = get_inference_time_and_accuracy(
                    model, test_dataloader, device
                )
                results.append([model_name, idx, new_name, avg_time, accuracy, size])

                if old_path != new_path:  # Ensure we are not renaming to the same name
                    os.rename(old_path, new_path)
                    print(
                        f"Renamed {file} to {new_name} in directory {model_directory}"
                    )

        except FileNotFoundError as e:
            print(e)
        except PermissionError:
            print(
                f"Permission denied when accessing directory or files in {model_directory}."
            )
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    # Save results to CSV file
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Model Name",
                "Model Number",
                "Model File",
                "Inference Time (s)",
                "Accuracy (Percentage)",
                "Model Size (bytes)",
            ]
        )
        writer.writerows(results)


if __name__ == "__main__":
    directory: str = "selected_models"
    models_name: List[str] = ["vit_b_16", "resnet50", "vgg16", "mobilenet_v3_large"]

    csv_file: str = "model_evaluation.csv"
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rename_models_and_evaluate(directory, models_name, csv_file, device)
