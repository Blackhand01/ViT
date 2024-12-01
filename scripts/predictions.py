"""
Contains functions for making predictions with the trained model.
"""

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from typing import List

def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str],
    transform: transforms.Compose = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """Predicts and plots the image with the predicted label."""
    model.eval()
    image = Image.open(image_path)

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    image_transformed = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(image_transformed)
        probs = torch.softmax(logits, dim=1)
        pred_label = probs.argmax(dim=1).item()

    plt.imshow(image)
    plt.title(f"Predicted: {class_names[pred_label]}")
    plt.axis("off")
    plt.show()
