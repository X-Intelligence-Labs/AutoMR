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
from AutoML_hyperparameter import get_best_hyperparameters

# Define a batch type for better clarity
Batch = Tuple[torch.Tensor, torch.Tensor]

# Instantiate the training dataset
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
    annotation_path='MHEALTH/cleaned_data/y_train.txt'  # Path to annotation file
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
    annotation_path='MHEALTH/cleaned_data/y_test.txt'  # Path to annotation file
)

# Read the dataset
mhealth_modality = mhealth_train.read_dataset(mhealth_train.data_path)

# Print available modalities to confirm the key values
print('Available modalities in the training dataset:', mhealth_modality.keys())

BATCH_SIZE = 256  # Set batch size

from torch.utils.data import Dataset, dataloader
# Instantiate DataAugmentor for data augmentation
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

    def augment_sequence(self, joints):
        if self.augmentor is not None:
            joints = self.augmentor.augment_data(joints)  # Apply the augment_data function
        return joints

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

# Use augmentor to enhance the dataset

# Seed the RNG for better reproducibility (see docs: https://pytorch.org/docs/stable/notes/randomness.html)
seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device type:', device)

# Load the data
X_train = mhealth_train.gesture_IMU
y_train = mhealth_train.label
X_test = mhealth_test.gesture_IMU
y_test = mhealth_test.label

# Dataset instance
train_set = MyDataset(X_train, y_train, augment=False, time_warp=False, augmentor=augmentor)
test_set = MyDataset(X_test, y_test, augment=False, time_warp=False, augmentor=augmentor)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

train_iter = iter(train_loader)
data, label = next(train_iter)

input_shape = data.shape
output_shape = mhealth_train.gesture_num

print('Input shape:', input_shape)
print('Output shape:', output_shape)

# Before creating the model, retrieve the best hyperparameters
model_name = 'quartznet'
dataset_name = 'MHEALTH'
best_config = get_best_hyperparameters(
    train_loader=train_loader,
    test_loader=test_loader,
    model_name=model_name,
    dataset_name=dataset_name,
    force_optimize=False  # Set to True to force re-optimization
)

print("Best hyperparameter configuration:", best_config)

# Use the best hyperparameters to create the model
path = f"./{model_name}_output/"
tb_logger = SummaryWriter(log_dir=path)

# Prepare model parameters
model_params = {
    "dropout_rate": best_config["dropout_rate"]
}

if model_name == "quartznet":
    model_params.update({
        "num_blocks": int(best_config["num_blocks"]),
        "num_cells": int(best_config["num_cells"]),
        "input_channels": int(best_config["input_channels"]),
        "head_channels": int(best_config["head_channels"]),
        "input_kernel": int(best_config["input_kernel"]),
        "head_kernel": int(best_config["head_kernel"])
    })

model = prepare_model(
    model_type=model_name,
    input_shape=input_shape,
    output_shape=output_shape,
    device=device,
    **model_params
)

# Set loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=best_config["learning_rate"],
    weight_decay=best_config["weight_decay"]
)

# Initialize evaluation metrics
train_accuracy = MulticlassAccuracy(device=device)
eval_accuracy = MulticlassAccuracy(device=device)
train_cm = MulticlassConfusionMatrix(num_classes=output_shape, device=device)
eval_cm = MulticlassConfusionMatrix(num_classes=output_shape, device=device)

# Create training unit
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

# Set up progress bar and garbage collector
progress_bar = TQDMProgressBar(refresh_rate=10)
garbage_collector = GarbageCollector(step_interval=51)

# Define callbacks (uncomment for additional functionality)
callbacks = [
    # progress_bar,
    garbage_collector,   
    # snapshot_saver,
]

# Start training
fit(
    my_unit,
    train_dataloader=train_loader,
    eval_dataloader=test_loader,
    max_epochs=400,
    callbacks=callbacks,
)

print("Training finished")
