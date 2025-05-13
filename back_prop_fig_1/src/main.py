#!/usr/bin/env python

from dataclasses import dataclass
from typing import Sequence

import torch
from model import SimmetryMLP
from torch import nn
from train_test import evaluate, train_model
from utils.environment import get_device, set_seed

# Create a DataLoader which creates those 64 possible sequences, try to not memorize (maybe do grooking on it also)
# You can check this with two pointers if they aren't working correctly break it (linear time)


# Define the dataclass for training arguments
@dataclass
class MLPTrainingArgs:
    batch_size: int = 128
    epochs: int = 5
    learning_rate: float = 1e-3
    hidden_sizes: Sequence[int] = ((64),)
    optimizer: str = "Adam"  # Default optimizer
    criterion: str = "CrossEntropyLoss"  # Default loss criterion
    seed: int = 42


def load_pretrained_model(args: MLPTrainingArgs, model_path: str, device: torch.device):
    """
    Load a trained model from the given path.
    """
    model = SimmetryMLP(hidden_sizes=args.hidden_sizes)

    # Ensure model_path is valid and remove invalid argument
    model.load_state_dict(torch.load(model_path, weights_only=True))

    return model.to(device)  # Move model to specified device


def main():
    # Instantiate training arguments based on parsed values
    args = MLPTrainingArgs()

    # Determine the best available device
    device = get_device()

    # Set random seed for reproducibility
    set_seed(args.seed)

    model = SimmetryMLP(hidden_sizes=args.hidden_sizes).to(device)

    # Initialize optimizer based on the argument
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    # Initialize criterion based on the argument
    criterion = nn.CrossEntropyLoss()

    train_loader, _ = get_dataloaders(args.batch_size)

    # model = train_model(args, device)


if __name__ == "__main__":
    main()
