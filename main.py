import torch
import torch.nn as nn
import torch.optim as optim
from func.seed import seed_everything
from func.get_data import get_dataloader
from model.ViTbase16 import ViTBase16
from run.train import train
from run.test import test
from func.func_for_plot import plot_loss, random_plot, plot_albumentation, plot_learning_rate
from func.get_scheduler import WarmupCosineAnnealingLR
import argparse
import os
import warnings
warnings.filterwarnings('ignore')


base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, r"checkpoint", r"best_model.pth")


def main(args):
    # Randomly plots images and their augmented versions for visualization
    random_plot()
    plot_albumentation()

    # Setting up the training environment based on provided arguments
    seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, valid_loader, test_loader = get_dataloader(args.batch_size)

    # Initializing the model, loss function, optimizer, and learning rate scheduler
    model = ViTBase16(n_classes=args.num_classes, pretrained=True).to(device)
    if args.test:
        # Load the best saved model and perform a test
        model.load_state_dict(torch.load(model_path))
        test_loss, test_accuracy = test(model, test_loader, nn.CrossEntropyLoss(), device=device)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    else:
        # Training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.warmup_start_lr)
        scheduler = WarmupCosineAnnealingLR(optimizer, args.warmup_epochs, args.epochs, args.warmup_start_lr,
                                            args.max_lr)

        # Training and evaluating the model
        trained_model, train_loss_history, valid_loss_history = train(model, train_loader, valid_loader, criterion,
                                                                      optimizer, scheduler=scheduler,
                                                                      epochs=args.epochs,
                                                                      device=device,
                                                                      early_stopping_patience=args.patience)

        # Plotting training and validation loss
        plot_loss(train_loss_history, valid_loss_history)

        # Perform a test
        test_loss, test_accuracy = test(trained_model, test_loader, criterion, device=device)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Vision Transformer on the Cassava dataset.")
    parser.add_argument("--seed", type=int, default=1001, help="Random seed for initialization")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation")
    parser.add_argument("--warmup_start_lr", type=float, default=1e-7, help="Starting learning rate for warmup")
    parser.add_argument("--max_lr", type=float, default=0.001, help="Maximum learning rate")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Number of warmup epochs before reaching max_lr")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes in the dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Total number of epochs for training")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--test", action='store_true', default=False,
                        help="Test the model using the best checkpoint without training.")

    args = parser.parse_args()
    main(args)

