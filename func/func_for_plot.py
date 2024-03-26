import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import pandas as pd
import json
from torchvision import transforms
import albumentations
from albumentations.pytorch import ToTensorV2
import math

base_dir = os.path.dirname(__file__)
IMG_SIZE = 224
image_path = os.path.join(base_dir, '..', r"image_output")
DATA_PATH = os.path.join(base_dir, '..', r"Datasets")
TRAIN_PATH = os.path.join(DATA_PATH, "train_images")
os.makedirs(image_path, exist_ok=True)
with open(os.path.join(DATA_PATH, "label_num_to_disease_map.json")) as f:
    label_map = json.load(f)


def random_plot():
    """
       Function random_plot:
       No input parameters.

       Output:
       Saves a PDF file containing a 3x3 grid of random images with their labels from the training dataset.

       Description:
       Selects nine random images from the training dataset and displays them in a 3x3 grid, saving the plot to a PDF.
    """
    df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    random_idx = np.random.randint(0, len(df), size=9)

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    for idx, ax in enumerate(axes.ravel()):
        img_name, label = df.values[random_idx[idx]]

        img_path = os.path.join(TRAIN_PATH, img_name)
        img = Image.open(img_path)
        ax.set_title(label_map[str(label)])
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(image_path, r"random_plot.pdf"), format='pdf')
    plt.close()



def plot_loss(train_loss_history, valid_loss_history, title='Training and Validation Loss'):
    """
        Function plot_loss:
        Input:
        - train_loss_history: List of training loss values.
        - valid_loss_history: List of validation loss values.
        - title: String, title of the plot (default is 'Training and Validation Loss').

        Output:
        Saves a PDF file with a plot of the training and validation loss histories.

        Description:
        Plots the training and validation loss over epochs and saves the plot to a PDF file.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(valid_loss_history, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(image_path, r"loss.pdf"), format='pdf')
    plt.close()


def plot_albumentation():
    """
        Function plot_albumentation:
        No input parameters.

        Output:
        Saves a PDF file showing a comparison between an original and augmented image.

        Description:
        Displays and saves a side-by-side comparison of an original image and its augmented version using albumentations library.
    """
    df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    random_idx = np.random.randint(0, len(df))
    img_name, label = df.values[random_idx]
    img_path = os.path.join(TRAIN_PATH, img_name)
    img = Image.open(img_path).convert("RGB")
    original_image = np.array(img)
    transforms_train = albumentations.Compose([
        albumentations.Resize(IMG_SIZE, IMG_SIZE),
        albumentations.Transpose(p=1),
        albumentations.HorizontalFlip(p=1),
        albumentations.VerticalFlip(p=1),
        albumentations.ShiftScaleRotate(p=1),
        albumentations.HueSaturationValue(10, 15, 10, p=1),
        albumentations.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1),
            contrast_limit=(-0.1, 0.1),
            p=1
        ),
        # albumentations.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225],
        #     max_pixel_value=255.0
        # ),
        albumentations.CoarseDropout(p=1),
        ToTensorV2(),

    ])
    augmented_image = transforms_train(image=original_image)['image']

    augmented_image = transforms.ToPILImage()(augmented_image)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original_image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(augmented_image)
    ax[1].set_title('Augmented Image')
    ax[1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(image_path, r"albumentation.pdf"), format='pdf')
    plt.close()


def plot_learning_rate(total_epochs=20, warmup_epochs=2, warmup_start_lr=1e-6, max_lr=0.001):
    """
        Function plot_learning_rate:
        Input:
        - total_epochs: Integer, total number of epochs (default is 20).
        - warmup_epochs: Integer, number of warmup epochs (default is 2).
        - warmup_start_lr: Float, starting learning rate for warmup (default is 1e-6).
        - max_lr: Float, maximum learning rate (default is 0.001).

        Output:
        Saves a PDF file with the learning rate schedule plot.

        Description:
        Plots the learning rate schedule across epochs, showing the warmup period and annealing phase.
    """
    end_lr = warmup_start_lr / 100

    def compute_lr(epoch):
        if epoch < warmup_epochs:
            lr = (max_lr - warmup_start_lr) / warmup_epochs * epoch + warmup_start_lr
        else:
            cos_out = math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)) + 1
            lr = (max_lr - end_lr) / 2 * cos_out + end_lr
        return lr

    lr_values = [compute_lr(epoch) for epoch in range(total_epochs)]

    plt.figure(figsize=(10, 6))
    plt.plot(lr_values, marker='o')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(image_path, r"lr.pdf"), format='pdf')
    plt.close()


if __name__ == '__main__':
    plot_albumentation()


