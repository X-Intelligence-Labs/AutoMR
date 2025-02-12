from typing import Literal, Optional, Tuple, Union
import torch
import torch.nn as nn
from dataset_module import Shrec2021Dataset, Modality
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
from dataset_module import GestureClass, SHREC_NAME, UCI_HAR_NAME

from dataset_module import UCI_HAR_dataset

import argparse
from model_factory_module import prepare_model
from train_module import TrainUnit
from augmentation import DataAugmentor
import numpy as np

# Define batch type for data and labels
Batch = Tuple[torch.Tensor, torch.Tensor]

# Instantiate the training dataset for UCI HAR
UCI_train = UCI_HAR_dataset(
    data_path='UCI_HAR/X_train.txt',  # Path to the training data
    dataset_name='UCI_HAR_train',  # Dataset name
    data_modality=[Modality.IMU],  # Modality type (IMU in this case)
    timestamps=None,
    split_joint_name=None,
    sensor_name='imu',  # Specify the sensor used
    sampling_rate=100.0,  # Sampling rate of the data
    padding_value=0,  # Padding value
    window_size=250,  # Window size for sliding window approach
    stride=64,  # Stride value
    annotation_path='UCI_HAR/y_train.txt'  # Path to the annotations for training
)

UCI_test = UCI_HAR_dataset(
    data_path='UCI_HAR/X_test.txt',  # Path to the test data
    dataset_name='UCI_HAR_test',  # Dataset name for test data
    data_modality=[Modality.IMU],  # Modality type (IMU)
    timestamps=None,
    split_joint_name=None,
    sensor_name='imu',  # Sensor name used for testing
    sampling_rate=100.0,
    padding_value=0,
    window_size=250,
    stride=64,
    annotation_path='UCI_HAR/y_test.txt'  # Path to the annotations for test
)

# Read the UCI dataset
UCI_modality = UCI_train.read_dataset(UCI_train.data_path)

# Print available modalities in the dataset
print('Available modalities in the training dataset:', UCI_modality.keys())

# Set batch size for data loading
BATCH_SIZE = 32

# Initialize data augmentation tool
augmentor = DataAugmentor()

# Custom Dataset class to apply augmentation and return formatted data
class MyDataset(Dataset):
    def __init__(self, data, labels, augment=False, time_warp=True, augmentor=None):
        self.data = data
        self.labels = labels
        self.augment = augment
        self.time_warp = time_warp
        self.augmentor = augmentor

    def __len__(self):
        return len(self.data)

    # Function to augment the sequence using the DataAugmentor
    def augment_sequence(self, joints):
        if self.augmentor is not None:
            joints = self.augmentor.augment_data(joints)  # Augment data
        return joints

    # Function to retrieve the data and apply augmentation if needed
    def __getitem__(self, idx):
        d = self.data[idx]  # Get the data at index
        if self.augment:
            d = self.augment_sequence(d)  # Apply data augmentation
        if self.time_warp:
            d = self.augmentor.time_warp(d)  # Apply time-warp augmentation
        d = np.reshape(d, (33, 17))  # Reshape the data to match model input
        return torch.from_numpy(d), self.labels[idx]

# Seed the random number generator for reproducibility
seed(0)

# Initialize the device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device type:', device)

# Instantiate datasets and dataloaders for training and testing
train_set = MyDataset(UCI_train.gesture_IMU, UCI_train.label, augment=False, time_warp=False, augmentor=augmentor)
test_set = MyDataset(UCI_test.gesture_IMU, UCI_test.label, augment=False, time_warp=False, augmentor=augmentor)

# Create DataLoader instances for batching
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

# Retrieve a batch of data to inspect the input and output shapes
train_iter = iter(train_loader)
data, label = next(train_iter)

# Get input and output shapes based on the first batch
input_shape = data.shape
output_shape = UCI_train.gesture_num

# Set model name and log path
model_name = 'quartznet'
path = f"./{model_name}_output/"
tb_logger = SummaryWriter(log_dir=path)

# Prepare the model based on input and output shapes
model = prepare_model(model_name, input_shape, output_shape, device)

# Define the loss function and evaluation metrics
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
    gesture_names=UCI_HAR_NAME  # Names of the gestures in the dataset
)

# Setup progress bar and garbage collector callbacks
progress_bar = TQDMProgressBar(refresh_rate=10)
garbage_collector = GarbageCollector(step_interval=51)

# Define the callbacks for the training process
callbacks = [
    # progress_bar,  # Uncomment for progress bar
    garbage_collector,   
    # snapshot_saver,  # Uncomment for model checkpoint saving
]

# Start the training process
fit(
    my_unit,
    train_dataloader=train_loader,
    eval_dataloader=test_loader,
    max_epochs=400,
    callbacks=callbacks,
)