import os

import torch
from model import SimpleMLP
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.data import get_dataloaders


def train(
    max_epochs: int,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int = 64,
) -> nn.Module:
    """
    Train a PyTorch model with optional activation recording at fractional epochs.
    """
    model.to(device)
    model.train()

    total_batches = len(train_loader)

    for epoch in range(1, max_epochs + 1):
        total_loss = 0.0
        progress_bar = tqdm(
            enumerate(train_loader, start=1),
            total=total_batches,
            desc=f"Epoch {epoch}/{max_epochs}",
            leave=False,
        )

        for batch_idx, (x, y) in progress_bar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / total_batches
        print(f"Epoch {epoch} completed - Avg Loss: {avg_loss:.4f}")

    return model


@torch.no_grad()
def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """
    Evaluate a PyTorch model on test data.
    """
    model.eval()
    model.to(device)

    correct, total = 0, 0

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        predicted = torch.argmax(output, dim=1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def train_model(args, device):
    """
    Initialize and train the model, optionally saving it and recording activations.
    """
    model = SimpleMLP().to(device)

    # Initialize optimizer based on the argument
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    # Initialize criterion based on the argument
    if args.criterion == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    elif args.criterion == "MSELoss":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported criterion: {args.criterion}")

    train_loader, _ = get_dataloaders(args.batch_size)

    # Train
    model = train(
        max_epochs=args.epochs,
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        batch_size=args.batch_size,
    )

    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model_path), exist_ok=True)
        torch.save(model.state_dict(), args.save_model_path)
        print(f"Model saved to {args.save_model_path}")

    return model
