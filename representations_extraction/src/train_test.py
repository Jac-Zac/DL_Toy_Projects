import os
from typing import Optional

import torch
from activation_hook import ActivationRecorder, save_model_activations
from model import initialize_model
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
    save_activation_frequency: float = 0.0,
    save_activation_path: str = "",
    batch_size: int = 64,
    recorder: Optional[ActivationRecorder] = None,
) -> nn.Module:
    """
    Train a PyTorch model with optional activation recording at fractional epochs.
    """
    model.to(device)
    model.train()

    total_batches = len(train_loader)
    batches_per_activation = (
        int(total_batches * save_activation_frequency)
        if save_activation_frequency
        else None
    )

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

            # Save activations at fractional intervals
            if (
                recorder
                and save_activation_frequency > 0
                and batches_per_activation
                and batch_idx % batches_per_activation == 0
            ):
                fractional_epoch = epoch - 1 + batch_idx / total_batches
                print(
                    f"Epoch {epoch} - Saving activations at fractional epoch {fractional_epoch:.2f}..."
                )
                save_model_activations(
                    save_activation_path=save_activation_path,
                    batch_size=batch_size,
                    model=model,
                    device=device,
                    epoch=f"{fractional_epoch:.2f}",
                    recorder=recorder,
                )

        avg_loss = total_loss / total_batches
        print(f"Epoch {epoch} completed - Avg Loss: {avg_loss:.4f}")

        # Save full-epoch activations if requested
        if (
            recorder
            and save_activation_frequency >= 1
            and epoch % int(save_activation_frequency) == 0
        ):
            print(f"Epoch {epoch} - Storing activations...")
            save_model_activations(
                save_activation_path=save_activation_path,
                batch_size=batch_size,
                model=model,
                device=device,
                epoch=epoch,
                recorder=recorder,
            )

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
    model, optimizer, criterion = initialize_model(args, device)
    train_loader, _ = get_dataloaders(args.batch_size)

    recorder = None
    if args.save_activation_frequency:
        recorder = ActivationRecorder(verbose=False)

        # Save initial activations before training
        save_model_activations(
            save_activation_path=args.save_activation_path,
            batch_size=args.batch_size,
            model=model,
            device=device,
            epoch=0,
            recorder=recorder,
        )

    # Train
    model = train(
        max_epochs=args.epochs,
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_activation_frequency=args.save_activation_frequency,
        save_activation_path=args.save_activation_path,
        batch_size=args.batch_size,
        recorder=recorder,
    )

    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model_path), exist_ok=True)
        torch.save(model.state_dict(), args.save_model_path)
        print(f"Model saved to {args.save_model_path}")

    return model
