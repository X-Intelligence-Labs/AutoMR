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
from dataset_module import GestureClass, BERKELEY_NAME, mhealth_dataset

import argparse
from model_factory_module import prepare_model
from train_module import TrainUnit
from augmentation import DataAugmentor
import numpy as np



# Instantiate the training dataset
Batch = Tuple[torch.Tensor, torch.Tensor]


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
    annotation_path='berkeley/clean_data/y_train.txt'  # 提供注释文件路径
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
    annotation_path='berkeley/clean_data/y_test.txt'  # 提供注释文件路径
)


# 读取数据
oppo_modality = berkeley_train.read_dataset(berkeley_train.data_path)


BATCH_SIZE = 256  # 设置批次大小

from torch.utils.data import Dataset, dataloader
# 实例化 DataAugmentor
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
            joints = self.augmentor.augment_data(joints)  # 调用 augment_data 函数
        return joints

    def __getitem__(self, idx):
        d = self.data[idx]  # 转为 numpy 数组进行数据增强
        if self.augment:
            d = self.augment_sequence(d)
        if self.time_warp:
            d = self.augmentor.time_warp(d)  # 调用 time_warp 函数进行时间变换

        return torch.from_numpy(d), self.labels[idx]
# 使用 augmentor 进行数据增强

# seed the RNG for better reproducibility. see docs https://pytorch.org/docs/stable/notes/randomness.html
seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device type:', device)


# 加载数据
X_train = berkeley_train.gesture_IMU
y_train = berkeley_train.label
X_test = berkeley_test.gesture_IMU
y_test = berkeley_test.label



###dataset instance
train_set = MyDataset(X_train, y_train, augment=False, time_warp=False, augmentor=augmentor)

test_set = MyDataset(X_test, y_test, augment=False, time_warp=False, augmentor=augmentor)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)


train_iter = iter(train_loader)
data, label = next(train_iter)

input_shape = data.shape
output_shape = len(BERKELEY_NAME)


print('input shape:', input_shape)
print('output shape:', output_shape)

model_name = 'quartznet'
path = f"./{model_name}_output/"
tb_logger = SummaryWriter(log_dir=path)
model = prepare_model(model_name, input_shape, output_shape, device)


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
    gesture_names=BERKELEY_NAME
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

print("finished")