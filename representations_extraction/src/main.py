#!/usr/bin/env python

from dataclasses import dataclass

from model import load_pretrained_model
from train_test import evaluate, train_model
from utils.args_parser import parse_args
from utils.data import get_dataloaders
from utils.environment import get_device, set_seed
from utils.plot import create_dash_app  # Import the create_dash_app function


# Define the dataclass for training arguments
@dataclass
class SimpleMLPTrainingArgs:
    batch_size: int = 128
    epochs: int = 5
    learning_rate: float = 1e-3
    optimizer: str = "Adam"  # Default optimizer
    criterion: str = "CrossEntropyLoss"  # Default loss criterion
    seed: int = 42
    save_model: bool = True  # Flag to save the model or not
    save_activation_frequency: float = 0  # Frequency of saving activations
    save_model_path: str = "./models/simple_mlp.pth"  # Path to save the model
    save_activation_path: str = "./activations/"  # Path to the saved activations


def main():
    # Parse command-line arguments using the imported function
    args = parse_args()

    # Instantiate training arguments based on parsed values
    training_args = SimpleMLPTrainingArgs(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        save_model_path=args.model_path,
        save_model=(args.mode == "train"),
        save_activation_frequency=args.save_activation_frequency,
        save_activation_path=args.activation_path,
    )

    # Determine the best available device
    device = get_device()

    # Set random seed for reproducibility
    set_seed(args.seed)

    if args.mode == "train":
        # Train the model and optionally save it
        print("Training the model...")
        model = train_model(training_args, device)
        # Optionally test the model
        _, test_loader = get_dataloaders(args.batch_size)
        evaluate(model, test_loader, device)
        return
    elif args.mode == "load":
        # Load pretrained model
        model = load_pretrained_model(args.model_path, device)
        _, test_loader = get_dataloaders(args.batch_size)
        evaluate(model, test_loader, device)

    # Now call the Dash app
    dash_app = create_dash_app(
        activation_dir=args.activation_path,  # Path to the activations directory
        epochs=args.epochs,  # The number of epochs
    )

    # Run the Dash app
    dash_app.run(debug=False)


if __name__ == "__main__":
    main()
