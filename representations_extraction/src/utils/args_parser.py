import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or load a pretrained SimpleMLP model."
    )

    # Mode: Train or load a pretrained model
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "load"],
        required=True,
        help="Mode: 'train' to train a new model or 'load' to load a pretrained model.",
    )

    # Path to save or load the model
    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/simple_mlp.pth",
        help="Path to save or load the model.",
    )

    # Batch size for training/testing
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for training/testing."
    )

    # Number of epochs for training
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs for training."
    )

    # Learning rate for training
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate for training."
    )

    # Random seed for reproducibility
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )

    # Frequency of saving activations (0 for never, 1 for every epoch, etc.)
    parser.add_argument(
        "--save-activation-frequency",
        type=float,
        default=0.5,
        help="Frequency of saving activations. 0 for never, 1 for every epoch, etc.",
    )

    # Path to save activations
    parser.add_argument(
        "--activation-path",
        type=str,
        default="./activations/",
        help="Path to save model activations.",
    )

    return parser.parse_args()
