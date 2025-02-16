from typing import Literal, Optional, Tuple, Union
import torch
import torch.nn as nn
from dataset_module import Modality, BERKELEY_NAME
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import MulticlassAccuracy, MulticlassConfusionMatrix
from torchtnt.framework.auto_unit import AutoUnit
from torchtnt.framework.fit import fit
from torchtnt.utils import seed
from torchtnt.framework.callbacks.garbage_collector import GarbageCollector
from torchtnt.framework.callbacks.tqdm_progress_bar import TQDMProgressBar

from dataset_module import mhealth_dataset
from model_factory_module import prepare_model
from train_module import TrainUnit
from augmentation import DataAugmentor
from AutoMR_hyperparameter import get_best_hyperparameters

import numpy as np

# Dataset preparation
berkeley_train = mhealth_dataset(
    data_path='berkeley/clean_data/X_train.npy', 
    dataset_name='berkeley_train',  
    data_modality=[Modality.IMU], 
    timestamps=None,
    split_joint_name=None,
    sensor_name='imu',
    sampling_rate=30, 
    padding_value=0,
    window_size=15,
    stride=7,
    annotation_path='berkeley/clean_data/y_train.txt'
)

berkeley_test = mhealth_dataset(
    data_path='berkeley/clean_data/X_test.npy', 
    dataset_name='berkeley_test',  
    data_modality=[Modality.IMU], 
    timestamps=None,
    split_joint_name=None,
    sensor_name='imu',
    sampling_rate=30, 
    padding_value=0,
    window_size=15,
    stride=7,
    annotation_path='berkeley/clean_data/y_test.txt'
)

# Read data
oppo_modality = berkeley_train.read_dataset(berkeley_train.data_path)

# Dataset class definition
class MyDataset(Dataset):
    def __init__(self, data, labels, augment=False, time_warp=False, augmentor=None):
        self.data = data
        self.labels = labels
        self.augment = augment
        self.time_warp = time_warp
        self.augmentor = augmentor

    def __len__(self):
        return len(self.data)

    def augment_sequence(self, joints):
        if self.augmentor is not None:
            joints = self.augmentor.augment_data(joints)
        return joints

    def __getitem__(self, idx):
        d = self.data[idx]
        if self.augment:
            d = self.augment_sequence(d)
        if self.time_warp:
            d = self.augmentor.time_warp(d)
        return torch.from_numpy(d), self.labels[idx]

def main():
    # Set random seed
    seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device type:', device)

    # Load data
    X_train = berkeley_train.gesture_IMU
    y_train = berkeley_train.label
    X_test = berkeley_test.gesture_IMU
    y_test = berkeley_test.label

    # Instantiate the dataset
    augmentor = DataAugmentor()
    train_set = MyDataset(X_train, y_train, augment=False, time_warp=False, augmentor=augmentor)
    test_set = MyDataset(X_test, y_test, augment=False, time_warp=False, augmentor=augmentor)

    # Initialize DataLoader (using temporary batch_size)
    temp_batch_size = 64
    train_loader = DataLoader(train_set, batch_size=temp_batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=temp_batch_size, shuffle=False)

    # Obtain the input, output shape
    train_iter = iter(train_loader)
    data, label = next(train_iter)
    input_shape = data.shape
    output_shape = len(BERKELEY_NAME)
    print('Input shape:', input_shape)
    print('Output shape:', output_shape)

    # Model configuration
    model_name = 'quartznet'
    dataset_name = 'BERKELEY'  # Add support for the Berkeley dataset

    # Get the best hyperparameters
    best_config = get_best_hyperparameters(
        train_loader=train_loader,
        test_loader=test_loader,
        model_name=model_name,
        dataset_name=dataset_name,
        force_optimize=False  # Set whether to force optimization again
    )

    # Recreate DataLoader using the best batch_size
    train_loader = DataLoader(train_set, batch_size=int(best_config['batch_size']), shuffle=True)
    test_loader = DataLoader(test_set, batch_size=int(best_config['batch_size']), shuffle=False)

    # Create model
    model_params = {
        "dropout_rate": best_config["dropout_rate"],
        "num_blocks": int(best_config["num_blocks"]),
        "num_cells": int(best_config["num_cells"]),
        "input_channels": int(best_config["input_channels"]),
        "head_channels": int(best_config["head_channels"]),
        "input_kernel": int(best_config["input_kernel"]),
        "head_kernel": int(best_config["head_kernel"])
    }

    # Set up tensorboard
    path = f"./berkeley/{model_name}_output_auto/"
    tb_logger = SummaryWriter(log_dir=path)
    
    # Prepare model
    model = prepare_model(model_name, input_shape, output_shape, device, **model_params)

    # Training configuration
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_config["learning_rate"],
        weight_decay=best_config["weight_decay"]
    )

    # Evaluation metrics
    train_accuracy = MulticlassAccuracy(device=device)
    eval_accuracy = MulticlassAccuracy(device=device)
    train_cm = MulticlassConfusionMatrix(num_classes=output_shape, device=device)
    eval_cm = MulticlassConfusionMatrix(num_classes=output_shape, device=device)

    # Training unit
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
        gesture_names=BERKELEY_NAME
    )

    # Callback functions
    callbacks = [
        GarbageCollector(step_interval=51),
    ]

    # Start training
    print("Starting training...")
    fit(
        my_unit,
        train_dataloader=train_loader,
        eval_dataloader=test_loader,
        max_epochs=400,
        callbacks=callbacks,
    )

    print("Training completed")

if __name__ == "__main__":
    main()
