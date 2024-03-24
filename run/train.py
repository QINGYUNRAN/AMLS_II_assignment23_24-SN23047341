import torch
from tqdm import tqdm
import copy
import os

base_dir = os.path.dirname(__file__)
Checkpoint_PATH = os.path.join(base_dir, '..', r"checkpoint")
os.makedirs(Checkpoint_PATH, exist_ok=True)
model_path = os.path.join(Checkpoint_PATH, 'best_model.pth')


def train(model, train_loader, valid_loader, criterion, optimizer, scheduler=None, epochs=20, device='cpu',
          early_stopping_patience=10):
    """
        Trains a model using the provided data loaders, loss criterion, optimizer, and learning rate scheduler.

        Inputs:
        - model: The neural network model to train.
        - train_loader: DataLoader for the training data.
        - valid_loader: DataLoader for the validation data.
        - criterion: Loss function used to calculate loss.
        - optimizer: Optimization algorithm.
        - scheduler (optional): Learning rate scheduler.
        - epochs (optional): Total number of training epochs.
        - device (optional): Device to run the training on, e.g., 'cpu' or 'cuda'.
        - early_stopping_patience (optional): Number of epochs to wait without improvement before stopping.

        Outputs:
        - model: The trained model with the best validation accuracy.
        - train_loss_history: List of average training loss per epoch.
        - valid_loss_history: List of average validation loss per epoch.

        Description:
        This function trains the model for a specified number of epochs, saving the model with the best validation accuracy.
        It supports early stopping if the validation accuracy does not improve for a given number of consecutive epochs.
    """
    best_val_accuracy = 0
    best_model_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    train_loss_history = []
    valid_loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        model.train()
        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss.item() / len(train_loader)
        train_loss_history.append(epoch_loss)
        if scheduler is not None:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss.item() / len(valid_loader)
        valid_loss_history.append(epoch_val_loss)
        print(
            f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )
        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print("Validation accuracy improved, saving model...")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation accuracy for {epochs_no_improve} epochs.")

            # Early stopping
        if epochs_no_improve >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    # Load the best model state before return
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), model_path)
    return model, train_loss_history, valid_loss_history
