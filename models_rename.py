import os
import csv
import time
import torch
from typing import List, Tuple, Any
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, GTSRB
import warnings

warnings.filterwarnings("ignore")


def get_inference_time_and_accuracy(
    model: torch.nn.Module, dataloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    model.to(device)
    correct: int = 0
    total: int = 0
    total_time: float = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            start_time = time.time()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_time += time.time() - start_time
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    average_time = total_time / total
    accuracy = 100 * correct / total
    torch.cuda.empty_cache()
    return average_time, accuracy


def rename_models_and_evaluate(
    directory: str,
    models_name: List[str],
    csv_file: str,
    device: torch.device,
    dataset_name: str = "cifar10",
) -> None:
    results: List[List[Any]] = []

    for model_name in models_name:
        if dataset_name == "cifar10":
            if "vit" in model_name:
                transform = transforms.Compose(
                    [
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                        ),
                    ]
                )
            else:
                transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                        ),
                    ]
                )

            test_dataset = CIFAR10(
                root="./data", train=False, download=True, transform=transform
            )
        elif dataset_name == "GTSRB":
            transform = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669]
                    ),
                ]
            )

            test_dataset = datasets.GTSRB(
                root="./data", split="test", download=True, transform=transform
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model_directory = os.path.join(directory, model_name)
        try:
            if not os.path.exists(model_directory) or not os.path.isdir(
                model_directory
            ):
                raise FileNotFoundError(
                    f"Directory {model_directory} does not exist or is not a directory."
                )

            files = [
                f
                for f in os.listdir(model_directory)
                if os.path.isfile(os.path.join(model_directory, f))
            ]

            if not files:
                raise FileNotFoundError(
                    f"No files found in directory {model_directory}."
                )

            model_files_with_times = []

            for file in files:
                old_path = os.path.join(model_directory, file)
                model = torch.load(old_path, map_location=device)
                try:
                    avg_time, accuracy = get_inference_time_and_accuracy(
                        model, test_dataloader, device
                    )
                    model_files_with_times.append((file, avg_time, accuracy))
                except Exception as e:
                    print(f"Error evaluating model {file}: {e}")
                    continue

            model_files_with_times.sort(key=lambda x: x[1], reverse=True)

            for idx, (file, avg_time, accuracy) in enumerate(
                model_files_with_times, start=1
            ):
                old_path = os.path.join(model_directory, file)
                file_extension = os.path.splitext(file)[1]
                new_name = f"{model_name}_{dataset_name}_{idx}{file_extension}"
                new_path = os.path.join(model_directory, new_name)

                results.append([model_name, idx, new_name, avg_time, accuracy])

                if old_path != new_path:
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

    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Model Name",
                "Model Number",
                "Model File",
                "Inference Time (s)",
                "Accuracy (Percentage)",
            ]
        )
        writer.writerows(results)


if __name__ == "__main__":
    directory = "selected_models"
    cifar_models_name = ["vit_b_16", "resnet50", "vgg16", "mobilenet_v3_large"]
    # gtsrb_models_name = ["alexnet"]

    csv_file = "model_information.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # rename_models_and_evaluate(directory, gtsrb_models_name, csv_file, device, "GTSRB")
    rename_models_and_evaluate(
        directory, cifar_models_name, csv_file, device, "cifar10"
    )
