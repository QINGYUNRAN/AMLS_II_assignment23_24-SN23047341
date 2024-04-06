# AMLS_II_assignment23_24-SN23047341
## Cassava Disease Classification Project

This project utilizes the Vision Transformer (ViT) model along with various image preprocessing techniques to classify Cassava images. The objective is to differentiate between different diseased and healthy Cassava plant images, addressing a multi-class image classification problem.

## Project Structure

Below is the directory layout for this project:

- `checkpoint/`: Directory for saving model checkpoints.
- `Datasets/`: Contains datasets and related files.
  - `train_images/`: Directory containing training images.
  - `label_num_to_disease_map.json`: JSON file mapping labels to disease names.
  - `train.csv`: CSV file with training data annotations.
- `func/`: Helper functions for various tasks.
  - `func_for_plot.py`: Functions for plotting results.
  - `get_data.py`: Functions to load and preprocess data.
  - `get_scheduler.py`: Functions to get the learning rate scheduler.
  - `seed.py`: Functions for setting the random seed.
- `image_output/`: Directory for output images such as plots and figures.
- `model/`: Vision Transformer models and related files.
  - `ViTbase16.py`: Variant of the ViT model with specific configurations.
- `run/`: Scripts to run the model training and testing.
  - `test.py`: Script for model testing.
  - `train.py`: Script for model training.
- `main.py`: Main script to run the entire pipeline.

## Main Script Arguments

In `main.py`, the following command-line arguments configure the training:

- `--seed`: Set the random seed for initialization (default: 1001).
- `--batch_size`: Specify the batch size for training and evaluation (default: 64).
- `--warmup_start_lr`: Set the starting learning rate for the warmup phase (default: 1e-7).
- `--max_lr`: Set the maximum learning rate (default: 0.001).
- `--warmup_epochs`: Define the number of warmup epochs before reaching `max_lr` (default: 2).
- `--num_classes`: Set the number of classes in the dataset (default: 5).
- `--epochs`: Set the total number of epochs for training (default: 20).
- `--patience`: Define the patience for early stopping (default: 10).


## Packages and Requirements

This program runs under Python version 3.10.
The project depends on several external libraries, which are listed in `environment.yml`. To install these dependencies, run the command below:
```sh
conda env create -f environment.yml
```

## Example Run Command

To start training with the default parameters, run the following command in the terminal:
```bash
python main.py
```
or you can choose your own parameters like this:
```bash
python main.py --seed 1001 --batch_size 64 --warmup_start_lr 1e-7 --max_lr 0.001 --warmup_epochs 2 --num_classes 5 --epochs 20 --patience 10
```

