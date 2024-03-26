import matplotlib.pyplot as plt
import torch
from PIL import Image
import os
import pandas as pd
from sklearn import model_selection
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.utils import resample
import albumentations
from albumentations.pytorch import ToTensorV2
import numpy as np

base_dir = os.path.dirname(__file__)
image_path = os.path.join(base_dir, '..', r"image_output")
os.makedirs(image_path, exist_ok=True)
DATA_PATH = os.path.join(base_dir, '..', r"Datasets")
TRAIN_PATH = os.path.join(DATA_PATH, "train_images")
TEST_PATH = os.path.join(DATA_PATH, "test_images")
IMG_SIZE = 224
transforms_train = albumentations.Compose([
        albumentations.RandomResizedCrop(IMG_SIZE, IMG_SIZE),
        albumentations.Transpose(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.ShiftScaleRotate(p=0.5),
        albumentations.HueSaturationValue(p=0.5),
        albumentations.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1),
            contrast_limit=(-0.1, 0.1),
            p=0.5
        ),

        albumentations.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0, p=1
                ),

        albumentations.CoarseDropout(p=0.5),
        ToTensorV2(),
    ])

transforms_valid = albumentations.Compose([
        albumentations.Resize(IMG_SIZE, IMG_SIZE),

        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])

transforms_test = albumentations.Compose([
        albumentations.Resize(IMG_SIZE, IMG_SIZE),

        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])


class CassavaDataset(Dataset):
    """Represents a dataset for Cassava Leaf Disease classification.

        Parameters:
        - df: DataFrame containing image file names and their labels.
        - data_path: The base directory where images are stored.
        - mode: Specifies the dataset mode, either 'train' or 'test'.
        - transforms: A set of transformations to apply to the images.

        Returns:
        - An instance of the dataset class with specified characteristics.
    """

    def __init__(self, df, data_path=DATA_PATH, mode="train", transforms=None):
        super().__init__()
        self.df_data = df.values
        self.data_path = data_path
        self.transforms = transforms
        self.mode = mode
        self.data_dir = "train_images" if mode == "train" else "test_images"

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, index):
        img_name, label = self.df_data[index]
        img_path = os.path.join(self.data_path, self.data_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        if self.transforms:
            img = self.transforms(image=np.array(img))['image']

        return img, label


def oversample(train_df):
    """Increases the number of samples of minority classes to balance the dataset.

        Parameters:
        - train_df: DataFrame containing the training data with potential class imbalance.

        Returns:
        - A new DataFrame where the class distribution is balanced through oversampling.
    """
    train_df = train_df.reset_index(drop=True)
    majority_class = train_df[train_df['label'] == train_df['label'].mode()[0]]
    minority_classes = train_df[train_df['label'] != train_df['label'].mode()[0]]
    majority_class_count = majority_class.shape[0]
    oversampled_list = [majority_class]
    for class_name, group in minority_classes.groupby('label'):
        oversampled_class = resample(group, replace=True, n_samples=majority_class_count, random_state=42)
        oversampled_list.append(oversampled_class)
    oversampled_df = pd.concat(oversampled_list)
    oversampled_df = oversampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return oversampled_df


def get_dataloader(batch_size):
    """Creates DataLoader instances for training, validation, and testing datasets.

        Parameters:
        - batch_size: The number of samples to load per batch.

        Returns:
        - Three DataLoader objects for the training, validation, and test datasets, respectively.
    """
    df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    print("The count of all data:", len(df))
    train_df, valid_test_df = model_selection.train_test_split(
        df, test_size=0.4, random_state=42, stratify=df.label.values
    )
    plt.figure()
    train_df.label.value_counts().plot(kind="bar")
    plt.savefig(os.path.join(image_path, "bar_plot.pdf"), format='pdf')
    plt.close()
    valid_df, test_df = model_selection.train_test_split(
        valid_test_df, test_size=0.5, random_state=42, stratify=valid_test_df.label.values
    )
    print("The original length of each dataset from train, val to test:", len(train_df), len(valid_df), len(test_df))
    train_df_resampled = oversample(train_df)
    plt.figure()
    train_df_resampled.label.value_counts().plot(kind="bar")
    plt.savefig(os.path.join(image_path, "balanced.pdf"), format='pdf')
    plt.close()
    train_dataset = CassavaDataset(train_df_resampled, transforms=transforms_train)
    valid_dataset = CassavaDataset(valid_df, transforms=transforms_valid)
    test_dataset = CassavaDataset(test_df, transforms=transforms_test)
    print("The oversampled length of each dataset from train, val to test:", len(train_dataset), len(valid_dataset), len(test_dataset))
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    get_dataloader(32)
