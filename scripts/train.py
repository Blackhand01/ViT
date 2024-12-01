"""
Trains the ViT model.
"""

import argparse
import torch
from torch import nn
from torchvision import transforms
from data_setup import create_dataloaders
from model_builder import ViT
from engine import train
from utils import set_seeds, plot_loss_curves, save_model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds()

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])

    train_dataloader, test_dataloader, class_names = create_dataloaders(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        transform=transform,
        batch_size=args.batch_size
    )

    model = ViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        num_classes=len(class_names),
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_size=args.mlp_size,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    results = train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=args.epochs,
        device=device
    )

    plot_loss_curves(results)

    save_model(
        model=model,
        target_dir=args.save_dir,
        model_name=args.model_name
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT model.")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training data.")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to testing data.")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save the model.")
    parser.add_argument("--model_name", type=str, default="vit_model.pth", help="Model filename.")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size.")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size.")
    parser.add_argument("--embedding_dim", type=int, default=768, help="Embedding dimension.")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of transformer layers.")
    parser.add_argument("--mlp_size", type=int, default=3072, help="MLP size.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--attn_dropout", type=float, default=0.0, help="Attention dropout rate.")
    parser.add_argument("--learning_rate", type=float, default=3e-3, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.3, help="Weight decay.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")

    args = parser.parse_args()
    main(args)
