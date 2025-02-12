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
from dataset_module import GestureClass, MHEALTH_NAME, mhealth_dataset

import argparse
from model_factory_module import prepare_model
from train_module import TrainUnit
from augmentation import DataAugmentor
import numpy as np

# Define the batch data type for better clarity
Batch = Tuple[torch.Tensor, torch.Tensor]

# Initialize the training dataset
mhealth_train = mhealth_dataset(
    data_path='MHEALTH/cleaned_data/X_train.npy', 
    dataset_name='mhealth_train',  
    data_modality=[Modality.EMG], 
    timestamps=None,
    split_joint_name=None,
    sensor_name='emg',
    sampling_rate=2000, 
    padding_value=0,
    window_size=520,
    stride=64,
    annotation_path='MHEALTH/cleaned_data/y_train.txt'  # Path to the annotation file
)

mhealth_test = mhealth_dataset(
    data_path='MHEALTH/cleaned_data/X_test.npy', 
    dataset_name='mhealth_test',  
    data_modality=[Modality.EMG], 
    timestamps=None,
    split_joint_name=None,
    sensor_name='emg',
    sampling_rate=2000, 
    padding_value=0,
    window_size=520,
    stride=64,
    annotation_path='MHEALTH/cleaned_data/y_test.txt'  # Path to the annotation file
)

# Read the dataset and retrieve the available modalities
mhealth_modality = mhealth_train.read_dataset(mhealth_train.data_path)

# Print available modalities to confirm key values
print('Available modalities in the training dataset:', mhealth_modality.keys())

BATCH_SIZE = 256  # Set the batch size

from torch.utils.data import Dataset, dataloader
# Instantiate the DataAugmentor for augmenting the dataset
augmentor = DataAugmentor()

class MyDataset(Dataset):
    def __init__(self, data, labels, augment=False, time_warp=True, augmentor=None):
        self.data = data
        self.labels = labels
        self.augment = augment
        self.time_warp = time_warp
        self.augmentor = augmentor

    def __len__(self):
        return len(self.data)

    # Apply data augmentation if specified
    def augment_sequence(self, joints):
        if self.augmentor is not None:
            joints = self.augmentor.augment_data(joints)  # Call the augment_data function
        return joints

    # Get a single item from the dataset
    def __getitem__(self, idx):
        d = self.data[idx]  # Convert to numpy array for data augmentation
        if self.augment:
            d = self.augment_sequence(d)
        if self.time_warp:
            d = self.augmentor.time_warp(d)  # Apply time_warp function for temporal transformation
        # Feature normalization (optional, currently commented out)
        # d_min = d.min(axis=(0,1))
        # d_max = d.max(axis=(0,1))
        # d_max = np.where(d_max == d_min, d_max + 1e-6, d_max)
        # d_norm = (d - d_min) / (d_max - d_min)
        # d_norm = np.reshape(d_norm, (130, 48))
        # mean = d.mean(axis=(0,1))
        # std = d.std(axis=(0,1))
        # d_norm = (d - mean) / std
        return torch.from_numpy(d), self.labels[idx]

# Use the augmentor to perform data augmentation

# Seed the random number generator for better reproducibility (see docs: https://pytorch.org/docs/stable/notes/randomness.html)
seed(0)

# Set the device to GPU if available, otherwise fallback to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device type:', device)

# Load the training and testing data
X_train = mhealth_train.gesture_IMU
y_train = mhealth_train.label
X_test = mhealth_test.gesture_IMU
y_test = mhealth_test.label

# Create dataset instances for training and testing
train_set = MyDataset(X_train, y_train, augment=False, time_warp=False, augmentor=augmentor)
test_set = MyDataset(X_test, y_test, augment=False, time_warp=False, augmentor=augmentor)

# Load the data into DataLoader for batching
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

# Get the input and output shapes from the dataset
train_iter = iter(train_loader)
data, label = next(train_iter)

input_shape = data.shape
output_shape = mhealth_train.gesture_num

# Print the input and output shapes
print('Input shape:', input_shape)
print('Output shape:', output_shape)

# Define the model name and create the TensorBoard logger
model_name = 'quartznet'
path = f"./{model_name}_output/"
tb_logger = SummaryWriter(log_dir=path)

# Prepare the model using the specified model name and hyperparameters
model = prepare_model(model_name, input_shape, output_shape, device)

# Define loss function (cross-entropy) and evaluation metrics
criterion = nn.CrossEntropyLoss()
train_accuracy = MulticlassAccuracy(device=device)
eval_accuracy = MulticlassAccuracy(device=device)
train_cm = MulticlassConfusionMatrix(num_classes=output_shape, device=device)
eval_cm = MulticlassConfusionMatrix(num_classes=output_shape, device=device)

# Initialize the training unit with relevant parameters
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
    gesture_names=MHEALTH_NAME
)

# Set up progress bar and garbage collector callbacks
progress_bar = TQDMProgressBar(refresh_rate=10)
garbage_collector = GarbageCollector(step_interval=51)

# List of callbacks (commented ones are optional)
callbacks = [
    # progress_bar,
    garbage_collector,   
    # snapshot_saver,
]

# Start training the model with the specified number of epochs
fit(
    my_unit,
    train_dataloader=train_loader,
    eval_dataloader=test_loader,
    max_epochs=400,
    callbacks=callbacks,
)

# Print a message after training is finished
print("Training finished")