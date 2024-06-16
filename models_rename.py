import os
from typing import List, Tuple


def rename_models(directory: str, models_name: List[str]) -> None:
    for model_name in models_name:
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

            # Rename files according to their size order from largest to smallest
            for idx, (file, _) in enumerate(model_files_with_sizes, start=1):
                old_path: str = os.path.join(model_directory, file)
                file_extension: str = os.path.splitext(file)[1]
                new_name: str = f"{model_name}_{idx}{file_extension}"
                new_path: str = os.path.join(model_directory, new_name)
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


directory: str = "selected_models"
models_name: List[str] = ["vit_b_16", "resnet50", "vgg16", "mobilenet_v3_large"]
rename_models(directory, models_name)
