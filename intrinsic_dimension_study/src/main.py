#!/usr/bin/env python

from dataclasses import dataclass

from train_test import evaluate, train_model
from utils.environment import get_device, set_seed


# Define the dataclass for training arguments
@dataclass
class MLPTrainingArgs:
    batch_size: int = 128
    epochs: int = 5
    learning_rate: float = 1e-3
    optimizer: str = "Adam"  # Default optimizer
    criterion: str = "CrossEntropyLoss"  # Default loss criterion
    seed: int = 42


def main():
    # Instantiate training arguments based on parsed values
    args = MLPTrainingArgs()

    # Determine the best available device
    device = get_device()

    # Set random seed for reproducibility
    set_seed(args.seed)

    print("Hello")

    # model = train_model(args, device)


if __name__ == "__main__":
    main()
