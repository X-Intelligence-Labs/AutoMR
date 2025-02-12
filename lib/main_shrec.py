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

from dataset_module import Shrec2021Dataset
import argparse
from model_factory_module import prepare_model
from train_module import TrainUnit
from augmentation import DataAugmentor
import numpy as np

# Define batch type for clarity
Batch = Tuple[torch.Tensor, torch.Tensor]

# Initialize the training dataset for Shrec2021
shrec_train = Shrec2021Dataset(
    data_path='shrec/shrec_train', 
    dataset_name='shrec2021_train',  
    data_modality=[Modality.JOINT], 
    timestamps=None,
    split_joint_name=None,
    sensor_name=None,
    sampling_rate=100.0, 
    padding_value=0,
    window_size=250,
    stride=64,
    annotation_path='shrec/annotation_shrec2021_train.txt'  # Path to the annotation file
)

shrec_test = Shrec2021Dataset(
    data_path='shrec/shrec_test', 
    dataset_name='shrec2021_test',  
    data_modality=[Modality.JOINT], 
    timestamps=None,
    split_joint_name=None,
    sensor_name=None,
    sampling_rate=100.0, 
    padding_value=0,
    window_size=250,
    stride=64,
    annotation_path='shrec/annotation_shrec2021_test.txt'  # Path to the annotation file
)

# Read the SHREC2021 dataset
shrec_modality = shrec_train.read_dataset(shrec_train.data_path)

# Print available modalities to verify keys
print('Available modalities in the training dataset:', shrec_modality.keys())

# Define batch size
BATCH_SIZE = 128

# Initialize data augmentation tools
augmentor = DataAugmentor()

# Custom dataset class that applies data augmentation
class MyDataset(Dataset):
    def __init__(self, data, labels, augment=False, time_warp=True, augmentor=None):
        self.data = data
        self.labels = labels
        self.augment = augment
        self.time_warp = time_warp
        self.augmentor = augmentor

    def __len__(self):
        return len(self.data)

    # Apply data augmentation to each sequence
    def augment_sequence(self, joints):
        if self.augmentor is not None:
            joints = self.augmentor.augment_data(joints)  # Apply augment_data function
        return joints

    # Retrieve each data sample and apply transformations
    def __getitem__(self, idx):
        d = self.data[idx]  # Convert to numpy array for augmentation
        if self.augment:
            d = self.augment_sequence(d)
        if self.time_warp:
            d = self.augmentor.time_warp(d)  # Apply time_warp function for temporal transformation
        return torch.tensor(d), self.labels[idx]

# Seed the random number generator for reproducibility
seed(0)

# Initialize the device for computation
device = init_from_env()
print('Device type:', device)

# Instantiate dataset and dataloaders for training and testing
train_set = MyDataset(shrec_train.gesture_data, shrec_train.label, augment=True, time_warp=True, augmentor=augmentor)
test_set = MyDataset(shrec_test.gesture_data, shrec_test.label, augment=False, time_warp=False, augmentor=augmentor)

# Create data loaders for batching
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

# Retrieve a batch of data to inspect the shapes
train_iter = iter(train_loader)
data, label = next(train_iter)

input_shape = data.shape
output_shape = shrec_train.gesture_num

print('Input shape:', input_shape)
print('Output shape:', output_shape)

# Model setup
model_name = 'quartznet'
path = f"./{model_name}_output/"
tb_logger = SummaryWriter(log_dir=path)

# Prepare the model based on the specified model name and input/output shapes
model = prepare_model(model_name, input_shape, output_shape, device)

# Loss function and evaluation metrics
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
    gesture_names=SHREC_NAME
)

# Progress bar and garbage collector setup
progress_bar = TQDMProgressBar(refresh_rate=10)
garbage_collector = GarbageCollector(step_interval=51)

# Define callbacks for the training process (commented-out options for additional functionality)
callbacks = [
    # progress_bar,
    garbage_collector,   
    # snapshot_saver,
]

# Start the training process with specified parameters
fit(
    my_unit,
    train_dataloader=train_loader,
    eval_dataloader=test_loader,
    max_epochs=400,
    callbacks=callbacks,
)