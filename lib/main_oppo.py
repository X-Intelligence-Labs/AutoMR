from typing import Literal, Optional, Tuple, Union
import torch
import torch.nn as nn
from dataset_module import Modality
from torch.utils.data import DataLoader
from torchtnt.utils.loggers import TensorBoardLogger
from torcheval.metrics import MulticlassAccuracy, MulticlassConfusionMatrix
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import MulticlassAccuracy, MulticlassConfusionMatrix, MulticlassF1Score
from torchtnt.framework.auto_unit import AutoUnit, Strategy, SWAParams, TrainStepResults
from torchtnt.framework.fit import fit
from torchtnt.framework.state import EntryPoint, State
from torchtnt.utils import init_from_env, seed, TLRScheduler
from torchtnt.utils.loggers import TensorBoardLogger
from torchtnt.utils.prepare_module import ActivationCheckpointParams, TorchCompileParams
from torchtnt.framework.callbacks.garbage_collector import GarbageCollector
from torchtnt.framework.callbacks.torchsnapshot_saver import TorchSnapshotSaver
from torchtnt.framework.callbacks.tqdm_progress_bar import TQDMProgressBar
from torchvision.utils import make_grid
import os
from dataset_module import GestureClass, OPPO_NAME, mhealth_dataset

import argparse
from model_factory_module import prepare_model
from train_module import TrainUnit
from augmentation import DataAugmentor
import numpy as np

# Define batch type for clarity
Batch = Tuple[torch.Tensor, torch.Tensor]

# Instantiate the training dataset for Oppo dataset
oppo_train = mhealth_dataset(
    data_path='oppo/clean_data/processed/s3X_train.npy', 
    dataset_name='oppo_train',  
    data_modality=[Modality.IMU], 
    timestamps=None,
    split_joint_name=None,
    sensor_name='imu',
    sampling_rate=30, 
    padding_value=0,
    window_size=15,
    stride=7,
    annotation_path='oppo/clean_data/processed/s3y_train.txt'  # Provide path to annotation file
)

oppo_test = mhealth_dataset(
    data_path='oppo/clean_data/processed/s3X_test.npy', 
    dataset_name='oppo_test',  
    data_modality=[Modality.IMU], 
    timestamps=None,
    split_joint_name=None,
    sensor_name='imu',
    sampling_rate=30, 
    padding_value=0,
    window_size=15,
    stride=7,
    annotation_path='oppo/clean_data/processed/s3y_test.txt'  # Provide path to annotation file
)

# Read dataset
oppo_modality = oppo_train.read_dataset(oppo_train.data_path)

# Print available modalities to confirm the correct key values
print('Available modalities in the training dataset:', oppo_modality.keys())

BATCH_SIZE = 256  # Set batch size for data loading

from torch.utils.data import Dataset, dataloader
# Instantiate DataAugmentor for potential data augmentation
augmentor = DataAugmentor()

class MyDataset(Dataset):
    def __init__(self, data, labels, augment=False, time_warp=True, augmentor=None):
        """
        Custom Dataset class for handling data and labels, with optional augmentation.
        """
        self.data = data
        self.labels = labels
        self.augment = augment
        self.time_warp = time_warp
        self.augmentor = augmentor

    def __len__(self):
        """
        Returns the length of the dataset (number of samples).
        """
        return len(self.data)

    def augment_sequence(self, joints):
        """
        Augment the input data sequence using the provided augmentor.
        """
        if self.augmentor is not None:
            joints = self.augmentor.augment_data(joints)  # Apply data augmentation
        return joints

    def __getitem__(self, idx):
        """
        Returns a sample (data, label) from the dataset.
        """
        d = self.data[idx]  # Retrieve the data sample
        if self.augment:
            d = self.augment_sequence(d)  # Apply augmentation if enabled
        if self.time_warp:
            d = self.augmentor.time_warp(d)  # Apply time warping transformation

        return torch.from_numpy(d), self.labels[idx]  # Convert to tensor and return

# Set random seed for reproducibility (see Pytorch randomness documentation)
seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device type:', device)

def remap_labels(labels):
    """
    Remaps the discontinuous labels to continuous labels starting from 0.
    
    Args:
        labels (np.ndarray): The original labels.

    Returns:
        np.ndarray: Remapped labels, and a dictionary of original-to-new label mapping.
    """
    unique_labels = sorted(np.unique(labels))  # Get sorted unique labels
    label_map = {old: new for new, old in enumerate(unique_labels)}  # Create label mapping
    remapped_labels = np.array([label_map[label] for label in labels])  # Map labels to continuous
    return remapped_labels

# Load training and test data
X_train = oppo_train.gesture_IMU
y_train = remap_labels(oppo_train.label)  # Remap labels for training
X_test = oppo_test.gesture_IMU
y_test = remap_labels(oppo_test.label)  # Remap labels for testing

# Create dataset instances for training and testing
train_set = MyDataset(X_train, y_train, augment=False, time_warp=False, augmentor=augmentor)
test_set = MyDataset(X_test, y_test, augment=False, time_warp=False, augmentor=augmentor)

# Create data loaders for batch processing
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

# Get a batch of data to inspect input and output shapes
train_iter = iter(train_loader)
data, label = next(train_iter)

input_shape = data.shape
output_shape = len(OPPO_NAME)

print('Input shape:', input_shape)
print('Output shape:', output_shape)

# Set model name and prepare model directory for logging
model_name = 'quartznet'
path = f"./{model_name}_output/"
tb_logger = SummaryWriter(log_dir=path)

# Initialize model with specified parameters
model = prepare_model(model_name, input_shape, output_shape, device)

# Define loss function and evaluation metrics
criterion = nn.CrossEntropyLoss()
train_accuracy = MulticlassAccuracy(device=device)
eval_accuracy = MulticlassAccuracy(device=device)
train_cm = MulticlassConfusionMatrix(num_classes=output_shape, device=device)
eval_cm = MulticlassConfusionMatrix(num_classes=output_shape, device=device)

# Set up training unit with the model, loss function, and metrics
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
    gesture_names=OPPO_NAME  # Define gesture names based on the dataset
)

# Define progress bar and garbage collector for efficient training
progress_bar = TQDMProgressBar(refresh_rate=10)
garbage_collector = GarbageCollector(step_interval=51)

# List of callbacks (uncomment if additional functionality is needed)
callbacks = [
    # progress_bar,
    garbage_collector,   
    # snapshot_saver,
]

# Start training process
fit(
    my_unit,
    train_dataloader=train_loader,
    eval_dataloader=test_loader,
    max_epochs=400,
    callbacks=callbacks,
)

print("Training finished")