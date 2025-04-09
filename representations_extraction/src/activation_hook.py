import os
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from utils.activations import save_activations_npz


class ActivationRecorder:
    def __init__(self, verbose: bool = False):
        self.activations = defaultdict(list)
        self.verbose = verbose
        self.hook_handles = []

    # Capture the layer name via a closure
    # Return the new hook with the specified layer name
    def _store_activation(self, layer_name):
        def hook(module, inputs, output):
            # Store the activation with that layer name
            # By appending it for each batch
            self.activations[layer_name].append(output.detach().cpu())
            if self.verbose:
                print(f"Stored activation for layer: {layer_name}")

        return hook

    def add_hooks(self, model: nn.Module) -> None:
        """
        Register hooks on all nn.Linear layers in the model.
        """
        for name, module in model.named_modules():
            # Get pre-activation
            # if isinstance(module, nn.ReLU):
            if isinstance(module, nn.Linear):
                # Include the type of layer in the layer name
                layer_name = f"{name} ({module})"
                # Passing a custom hook function for that specific layer, with its name already baked in.
                handle = module.register_forward_hook(
                    self._store_activation(layer_name)
                )
                self.hook_handles.append(handle)

    def remove_hooks(self) -> None:
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary mapping layer names to a list of flattened batched activation.
        """
        return {k: torch.cat(v, dim=0) for k, v in self.activations.items()}

    def clear(self) -> None:
        """
        Clear all stored activations.
        """
        self.activations.clear()


def record_activations(
    model: torch.nn.Module,
    recorder: ActivationRecorder,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Records layer activations from the model for all examples in the dataset.

    Returns:
        Tuple containing:
            - Dictionary mapping layer names to activation arrays.
            - Numpy array of corresponding labels.
    """
    model.eval()
    recorder.clear()
    recorder.add_hooks(model)

    activations: Dict[str, list[torch.Tensor]] = defaultdict(list)
    labels: list[int] = []

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            _ = model(x)
            labels.extend(y.cpu().numpy())

            # Append the current batch activations for each layer
            for layer_name, acts in recorder.get_activations().items():
                current_batch = acts[-x.size(0) :]  # Get the latest batch
                activations[layer_name].append(current_batch)

    recorder.remove_hooks()

    # Concatenate across batches and convert to NumPy
    final_activations = {
        layer: torch.cat(batches, dim=0).numpy()
        for layer, batches in activations.items()
    }
    final_labels = np.array(labels)

    return final_activations, final_labels


def save_model_activations(
    save_activation_path: str,
    model: torch.nn.Module,
    device: torch.device,
    recorder: ActivationRecorder,
    epoch: str | float | int,
    batch_size: int = 32,
) -> None:
    """
    Record activations for both train and test sets and save them to separate folders.
    """
    from utils.data import get_dataloaders

    # Define subdirectories for train and test
    train_path = os.path.join(save_activation_path, "train")
    test_path = os.path.join(save_activation_path, "test")

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # Get data loaders
    train_loader, test_loader = get_dataloaders(
        batch_size, shuffle=False, num_elements=384
    )

    # Record and save activations for train set
    train_activations, train_labels = record_activations(
        model, recorder, train_loader, device
    )

    # Create an epoch string
    epoch_str = f"{float(epoch):05.2f}".replace(".", "_")

    save_activations_npz(
        os.path.join(save_activation_path, "train"),
        f"epoch_{epoch_str}.npz",
        train_activations,
        train_labels,
    )
    # Record and save activations for test set
    test_activations, test_labels = record_activations(
        model, recorder, test_loader, device
    )

    save_activations_npz(
        os.path.join(save_activation_path, "test"),
        f"epoch_{epoch_str}.npz",
        test_activations,
        test_labels,
    )

    # Clean up
    recorder.remove_hooks()
