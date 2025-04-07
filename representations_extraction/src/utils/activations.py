import os
from typing import Dict, List, Union

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


def load_activations(activation_dir: str, epochs: Union[int, List[float], List[str]]):
    """
    Load activations and labels for train and test from the stored .npz files
    located in separate subdirectories.

    Args:
        activation_dir: Root directory containing 'train' and 'test' subfolders
        epochs: Number of epochs to load (int), or a list of specific epochs (float or str)

    Returns:
        Dictionary containing train and test activations organized by epoch and layer,
        along with their corresponding labels
    """
    train_activations_by_epoch = {}
    test_activations_by_epoch = {}
    train_labels = None
    test_labels = None

    if isinstance(epochs, int):
        epoch_list = [f"{i:05.2f}" for i in range(epochs)]
    else:
        epoch_list = [f"{float(e):05.2f}" for e in epochs]

    for epoch_str in epoch_list:
        # Format filename (underscore for decimal point to match saved file naming convention)
        epoch_filename = f"epoch_{epoch_str.replace('.', '_')}.npz"

        train_file = os.path.join(activation_dir, "train", epoch_filename)
        test_file = os.path.join(activation_dir, "test", epoch_filename)

        # Load the npz files
        train_data = np.load(train_file, allow_pickle=True)
        test_data = np.load(test_file, allow_pickle=True)

        # Convert string back to float for consistent epoch key
        epoch_key = float(epoch_str)

        # Store labels (assuming they're consistent across epochs)
        if train_labels is None:
            train_labels = train_data["labels"]
        if test_labels is None:
            test_labels = test_data["labels"]

        # Initialize the dictionary for this epoch
        train_activations_by_epoch[epoch_key] = {}
        test_activations_by_epoch[epoch_key] = {}

        # Store activations by layer for this epoch
        for key in train_data.keys():
            if key != "labels":
                train_activations_by_epoch[epoch_key][key] = train_data[key]

        for key in test_data.keys():
            if key != "labels":
                test_activations_by_epoch[epoch_key][key] = test_data[key]

    return {
        "train": {"activations": train_activations_by_epoch, "labels": train_labels},
        "test": {"activations": test_activations_by_epoch, "labels": test_labels},
    }
