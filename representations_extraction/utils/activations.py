import os
from typing import Dict

import numpy as np


def save_activations_npz(
    save_path: str,
    filename: str,
    activations: Dict[str, np.ndarray],
    labels: np.ndarray,
) -> None:
    """
    Saves activations and labels into a compressed `.npz` file.
    """
    data_to_save = {f"act_{layer}": acts for layer, acts in activations.items()}
    data_to_save["labels"] = labels
    np.savez_compressed(os.path.join(save_path, filename), **data_to_save)


def load_activations(activation_dir: str, epochs: int):
    """
    Load activations and labels for train and test from the stored .npz files.

    Args:
        activation_dir: Directory containing the activation .npz files
        epochs: Number of epochs to load

    Returns:
        Dictionary containing train and test activations organized by epoch and layer,
        along with their corresponding labels
    """
    train_activations_by_epoch = {}
    test_activations_by_epoch = {}
    train_labels = None
    test_labels = None

    for epoch in range(epochs):
        train_file = os.path.join(activation_dir, f"train_activations_epoch{epoch}.npz")
        test_file = os.path.join(activation_dir, f"test_activations_epoch{epoch}.npz")

        # Load the npz files
        train_data = np.load(train_file, allow_pickle=True)
        test_data = np.load(test_file, allow_pickle=True)

        # Store labels (assuming they're consistent across epochs)
        if train_labels is None:
            train_labels = train_data["labels"]
        if test_labels is None:
            test_labels = test_data["labels"]

        # Initialize the dictionary for this epoch
        train_activations_by_epoch[epoch] = {}
        test_activations_by_epoch[epoch] = {}

        # Store activations by layer for this epoch
        for key in train_data.keys():
            if key != "labels":
                train_activations_by_epoch[epoch][key] = train_data[key]

        for key in test_data.keys():
            if key != "labels":
                test_activations_by_epoch[epoch][key] = test_data[key]

    return {
        "train": {"activations": train_activations_by_epoch, "labels": train_labels},
        "test": {"activations": test_activations_by_epoch, "labels": test_labels},
    }
