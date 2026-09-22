"""Project configuration."""

import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_device_info() -> None:
    """Print the device used for training."""

    print(f"Using device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")