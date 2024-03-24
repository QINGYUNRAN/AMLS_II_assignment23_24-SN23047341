import torch


def test(model, test_loader, criterion, device='cpu'):
    """
        Evaluates the model's performance on a test dataset.

        Inputs:
        - model: The neural network model to be evaluated.
        - test_loader: DataLoader for the test dataset.
        - criterion: The loss function used for evaluation.
        - device: The device ('cpu' or 'cuda') on which to perform the evaluation.

        Outputs:
        - test_loss: The average loss of the model on the test dataset.
        - test_accuracy: The accuracy of the model on the test dataset.
    """
    model.eval()
    test_loss = 0
    test_accuracy = 0

    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            test_loss += loss.item() * data.size(0)
            test_accuracy += (output.argmax(dim=1) == label).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy /= len(test_loader.dataset)

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    return test_loss, test_accuracy
