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
from dataset_module import GestureClass, DB4_NAME, DB4_dataset

import argparse
from model_factory_module import prepare_model
from train_module import TrainUnit
from augmentation import DataAugmentor
import numpy as np

# Instantiate the training dataset
Batch = Tuple[torch.Tensor, torch.Tensor]

db4_train = DB4_dataset(
    data_path='db4/X_train.txt', 
    dataset_name='db4_train',  
    data_modality=[Modality.EMG], 
    timestamps=None,
    split_joint_name=None,
    sensor_name='emg',
    sampling_rate=2000, 
    padding_value=0,
    window_size=520,
    stride=64,
    annotation_path='db4/y_train.txt'  # Provide annotation file path
)

db4_test = DB4_dataset(
    data_path='db4/X_test.txt', 
    dataset_name='db4_test',  
    data_modality=[Modality.EMG], 
    timestamps=None,
    split_joint_name=None,
    sensor_name='emg',
    sampling_rate=2000, 
    padding_value=0,
    window_size=520,
    stride=64,
    annotation_path='db4/y_test.txt'  # Provide annotation file path
)

# Read the dataset
db4_modality = db4_train.read_dataset(db4_train.data_path)

# Print available modalities to verify keys
print('Available modalities in the training dataset:', db4_modality.keys())

BATCH_SIZE = 256  # Set batch size

from torch.utils.data import Dataset, dataloader
# Instantiate the DataAugmentor
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
            joints = self.augmentor.augment_data(joints)  # Call augment_data function
        return joints

    def __getitem__(self, idx):
        d = self.data[idx]  # Convert to numpy array for data augmentation
        if self.augment:
            d = self.augment_sequence(d)
        if self.time_warp:
            d = self.augmentor.time_warp(d)  # Call time_warp function for time transformation
        d = np.reshape(d, (520, 12))  # Align shape for feature normalization
        # Compute mean and std for each channel (across samples and time_steps)
        # d_min = d.min(axis=(0,1))
        # d_max = d.max(axis=(0,1))
        # d_max = np.where(d_max == d_min, d_max + 1e-6, d_max)
        # d_norm = (d - d_min) / (d_max - d_min)
        # d_norm = np.reshape(d_norm, (130, 48))
        # mean = d.mean(axis=(0,1))
        # std = d.std(axis=(0,1))
        # d_norm = (d - mean) / std
        return torch.from_numpy(d), self.labels[idx]
# Use augmentor for data augmentation

# Seed the RNG for better reproducibility. See docs https://pytorch.org/docs/stable/notes/randomness.html
seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device type:', device)

# Load data
X_train = db4_train.gesture_EMG
y_train = db4_train.label
X_test = db4_test.gesture_EMG
y_test = db4_test.label

# Function: Randomly sample a specified ratio of data to address class imbalance, here REST class is over 50 times larger than other classes, so we take 2% of REST data
def sample_data(X, y, label, ratio):
    # Select indices of the target class (label 0)
    target_indices = np.where(y == label)[0]
    other_indices = np.where(y != label)[0]
    
    # Calculate number of samples to be taken
    sample_size = int(len(target_indices) * ratio)
    
    # Perform random sampling
    sampled_indices = np.random.choice(target_indices, sample_size, replace=False)
    
    # Merge sampled indices with the other class indices
    final_indices = np.concatenate([sampled_indices, other_indices])
    
    # Shuffle the order
    np.random.shuffle(final_indices)
    
    # Return the re-ordered data
    return X[final_indices], y[final_indices]

# Balance the training set
X_train_balance, y_train_balance = sample_data(X_train, y_train, label=0, ratio=0.02)

# Balance the test set
X_test_balance, y_test_balance = sample_data(X_test, y_test, label=0, ratio=0.02)

### Dataset instance
train_set = MyDataset(X_train_balance, y_train_balance, augment=False, time_warp=False, augmentor=augmentor)

test_set = MyDataset(X_test_balance, y_test_balance, augment=False, time_warp=False, augmentor=augmentor)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

train_iter = iter(train_loader)
data, label = next(train_iter)

input_shape = data.shape
output_shape = db4_train.gesture_num

print('Input shape:', input_shape)
print('Output shape:', output_shape)

model_name = 'quartznet'
path = f"./{model_name}_output/"
tb_logger = SummaryWriter(log_dir=path)
model = prepare_model(model_name, input_shape, output_shape, device)
# model = prepare_model("shallow_resnet", input_shape, output_shape, device)

criterion = nn.CrossEntropyLoss()
train_accuracy = MulticlassAccuracy(device=device)
eval_accuracy = MulticlassAccuracy(device=device)
train_cm = MulticlassConfusionMatrix(num_classes=output_shape, device=device)
eval_cm = MulticlassConfusionMatrix(num_classes=output_shape, device=device)

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
    gesture_names=DB4_NAME
)

progress_bar = TQDMProgressBar(refresh_rate=10)
garbage_collector = GarbageCollector(step_interval=51)
# snapshot_saver = TorchSnapshotSaver(dirpath="./output/snapshots/", save_every_n_epochs=5, keep_last_n_checkpoints=2)

callbacks = [
    # progress_bar,
    garbage_collector,   
    # snapshot_saver,
]

fit(
    my_unit,
    train_dataloader=train_loader,
    eval_dataloader=test_loader,
    max_epochs=400,
    callbacks=callbacks,
)

print("Finished")