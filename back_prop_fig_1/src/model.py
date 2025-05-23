from typing import Callable, Sequence

import torch.nn as nn


# So create a simple netwrok which takes numbers as bit (8 bit) and returns 1 if simmetric -1 if not for example
# Perhaps test if there is any difference in training it with 1 and 0
class SimmetryMLP(nn.Module):
    """
    A simple, flexible Multi-Layer Perceptron (MLP).

    Args:
        input_size: Size of the input layer.
        hidden_sizes: Sizes of the hidden layers.
        num_classes: Size of the output layer.
        activation: Activation function to apply between layers.
            Should be a callable that returns an instance of an activation function (e.g., nn.ReLU).
    """

    def __init__(
        self,
        input_size: int = 8,
        hidden_sizes: Sequence[int] = (256, 64),
        num_classes: int = 1,
        activation: Callable[[], nn.Module] = nn.ReLU,
    ):
        super().__init__()

        # Validate inputs
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValueError("Input size must be a positive integer")
        if not all(isinstance(n, int) and n > 0 for n in hidden_sizes):
            raise ValueError("All hidden layer sizes must be positive integers")
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("Number of classes must be a positive integer")

        # Build hidden layers
        self.hidden_layers_list = []

        prev_size = input_size
        for h_size in hidden_sizes:
            # `activation()` returns an instance of an activation function
            self.hidden_layers_list.extend((nn.Linear(prev_size, h_size), activation()))
            prev_size = h_size

        # Create the sequential network for hidden layers
        self.hidden_layers = nn.Sequential(*self.hidden_layers_list)

        # Output layer
        self.output_layer = nn.Linear(prev_size, num_classes)

        # Softmax the output
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        # Return logits
        return self.softmax(x)
