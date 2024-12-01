"""
Contains utility functions.
"""

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

def set_seeds(seed=42):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training and validation loss and accuracy curves."""
    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, results["train_loss"], label="Train Loss")
    plt.plot(epochs, results["test_loss"], label="Test Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, results["train_acc"], label="Train Accuracy")
    plt.plot(epochs, results["test_acc"], label="Test Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    plt.show()

def save_model(
    model: torch.nn.Module,
    target_dir: str,
    model_name: str
):
    """Saves the model to the target directory."""
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), \
        "model_name should end with '.pth' or '.pt'"

    model_save_path = target_dir_path / model_name
    print(f"Saving model to: {model_save_path}")
    torch.save(model.state_dict(), model_save_path)
