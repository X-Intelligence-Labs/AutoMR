import os
from typing import Literal, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import MulticlassAccuracy, MulticlassConfusionMatrix
from torchtnt.framework.fit import fit
from torchtnt.framework.callbacks.garbage_collector import GarbageCollector
from torchtnt.framework.callbacks.tqdm_progress_bar import TQDMProgressBar
from torchtnt.utils import seed
from dataset_module import Modality, OPPO_NAME, mhealth_dataset
from model_factory_module import prepare_model
from train_module import TrainUnit
from augmentation import DataAugmentor
from AutoML_hyperparameter import get_best_hyperparameters

# Define the Batch type for better clarity
Batch = Tuple[torch.Tensor, torch.Tensor]

# Instantiate the training and testing datasets
oppo_train = mhealth_dataset(
    data_path='oppo/clean_data/processed/s4X_train.npy', 
    dataset_name='oppo_train',  
    data_modality=[Modality.IMU], 
    timestamps=None,
    split_joint_name=None,
    sensor_name='imu',
    sampling_rate=30, 
    padding_value=0,
    window_size=15,
    stride=7,
    annotation_path='oppo/clean_data/processed/s4y_train.txt'  # Path to the annotation file for training
)

oppo_test = mhealth_dataset(
    data_path='oppo/clean_data/processed/s4X_test.npy', 
    dataset_name='oppo_test',  
    data_modality=[Modality.IMU], 
    timestamps=None,
    split_joint_name=None,
    sensor_name='imu',
    sampling_rate=30, 
    padding_value=0,
    window_size=15,
    stride=7,
    annotation_path='oppo/clean_data/processed/s4y_test.txt'  # Path to the annotation file for testing
)

# Data Augmentor instance for data enhancement
augmentor = DataAugmentor()

# Define the dataset class to handle data loading and augmentation
class MyDataset(Dataset):
    def __init__(self, data, labels, augment=False, time_warp=False, augmentor=None):
        self.data = data  # Data samples
        self.labels = labels  # Corresponding labels
        self.augment = augment  # Flag to enable/disable data augmentation
        self.time_warp = time_warp  # Flag to enable/disable time warping
        self.augmentor = augmentor  # Instance of DataAugmentor

    def __len__(self):
        return len(self.data)  # Return the number of samples in the dataset

    def augment_sequence(self, joints):
        if self.augmentor is not None:
            joints = self.augmentor.augment_data(joints)  # Apply data augmentation if augmentor is provided
        return joints

    def __getitem__(self, idx):
        d = self.data[idx]  # Get the data sample
        if self.augment:
            d = self.augment_sequence(d)  # Apply augmentation if enabled
        if self.time_warp:
            d = self.augmentor.time_warp(d)  # Apply time warping if enabled
        return torch.from_numpy(d), self.labels[idx]  # Return data sample and label

# Function to remap the labels for classification
def remap_labels(labels):
    unique_labels = sorted(np.unique(labels))  # Get sorted unique labels
    label_map = {old: new for new, old in enumerate(unique_labels)}  # Create a mapping from old to new labels
    remapped_labels = np.array([label_map[label] for label in labels])  # Apply the label mapping
    return remapped_labels

# Main function to initiate the training process
def main():
    seed(0)  # Set random seed for reproducibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Choose device (GPU/CPU)
    print('Using device:', device)

    # Load training and testing data
    X_train = oppo_train.gesture_IMU
    y_train = remap_labels(oppo_train.label)  # Remap training labels
    X_test = oppo_test.gesture_IMU
    y_test = remap_labels(oppo_test.label)  # Remap testing labels

    # Create dataset instances for training and testing
    BATCH_SIZE = 256
    train_set = MyDataset(X_train, y_train, augment=False, time_warp=False, augmentor=augmentor)
    test_set = MyDataset(X_test, y_test, augment=False, time_warp=False, augmentor=augmentor)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)  # Create DataLoader for training
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)  # Create DataLoader for testing

    # Print input and output shapes to verify the data dimensions
    input_shape = next(iter(train_loader))[0].shape
    output_shape = len(OPPO_NAME)  # Number of classes (gesture names)
    print('Input shape:', input_shape)
    print('Output shape:', output_shape)

    # Model configuration: Get the best hyperparameters for the model
    model_name = 'quartznet'
    best_config = get_best_hyperparameters(train_loader, test_loader, model_name, "OPPO")  # Retrieve optimal hyperparameters

    # Prepare the model and TensorBoard logger
    path = f"./{model_name}_output_auto/"
    tb_logger = SummaryWriter(log_dir=path)
    model = prepare_model(model_name, input_shape, output_shape, device)  # Initialize the model

    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_config["learning_rate"],  # Learning rate from best hyperparameters
        weight_decay=best_config["weight_decay"]  # Weight decay for regularization
    )

    # Initialize evaluation metrics for training and testing
    train_accuracy = MulticlassAccuracy(device=device)
    eval_accuracy = MulticlassAccuracy(device=device)
    train_cm = MulticlassConfusionMatrix(num_classes=output_shape, device=device)
    eval_cm = MulticlassConfusionMatrix(num_classes=output_shape, device=device)

    # Create the training unit
    my_unit = TrainUnit(
        tb_logger=tb_logger,
        train_accuracy=train_accuracy,
        train_cm=train_cm,
        eval_accuracy=eval_accuracy,
        eval_cm=eval_cm,
        model_name=model_name,
        module=model,
        criterion=criterion,
        device=device,
        strategy=None,
        log_every_n_steps=10,
        gradient_accumulation_steps=1,
        detect_anomaly=True,
        clip_grad_norm=1.0,
        gesture_names=OPPO_NAME  # Use the gesture names from OPPO dataset
    )

    # Define callbacks (e.g., for progress tracking and memory management)
    callbacks = [
        TQDMProgressBar(refresh_rate=10),  # Show progress bar
        GarbageCollector(step_interval=51),  # Clean up memory periodically
    ]

    # Start the training process
    fit(
        my_unit,
        train_dataloader=train_loader,
        eval_dataloader=test_loader,
        max_epochs=400,  # Set maximum number of training epochs
        callbacks=callbacks,  # Include the defined callbacks
    )

    print("Training complete")  # Training finished

# Entry point for running the script
if __name__ == "__main__":
    main()  # Execute the main function to start training
