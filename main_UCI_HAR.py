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



# Instantiate the training dataset
Batch = Tuple[torch.Tensor, torch.Tensor]


UCI_train = UCI_HAR_dataset(
    data_path='UCI_HAR/X_train.txt', 
    dataset_name='UCI_HAR_train',  
    data_modality=[Modality.IMU], 
    timestamps=None,
    split_joint_name=None,
    sensor_name='imu',
    sampling_rate=100.0, 
    padding_value=0,
    window_size=250,
    stride=64,
    annotation_path='UCI_HAR/y_train.txt'  # 提供注释文件路径
)

UCI_test = UCI_HAR_dataset(
    data_path='UCI_HAR/X_test.txt', 
    dataset_name='UCI_HAR_test',  
    data_modality=[Modality.IMU], 
    timestamps=None,
    split_joint_name=None,
    sensor_name='imu',
    sampling_rate=100.0, 
    padding_value=0,
    window_size=250,
    stride=64,
    annotation_path='UCI_HAR/y_test.txt'  # 提供注释文件路径
)


# 读取数据
UCI_modality = UCI_train.read_dataset(UCI_train.data_path)

# 打印可用的模态以确认键值
print('Available modalities in the training dataset:', UCI_modality.keys())

# 获取相应的注释和标签
# train_joint_data, train_label = shrec_train.gesture_data, shrec_train.labels
BATCH_SIZE = 32  # 设置批次大小

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
        d = np.reshape(d,(33, 17))  # Shape allignment for model implement
        return torch.from_numpy(d), self.labels[idx]
# 使用 augmentor 进行数据增强

# seed the RNG for better reproducibility. see docs https://pytorch.org/docs/stable/notes/randomness.html
seed(0)

# device and process group initialization, gpu incompaitible:
# device = init_from_env()
# device = torch.device('cpu')
# print('device type---------------------------------', device)
# device and process group initialization, gpu incompaitible:
# device = init_from_env()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device type---------------------------------', device)

###dataset instance
train_set = MyDataset(UCI_train.gesture_IMU, UCI_train.label, augment=False, time_warp=False, augmentor=augmentor)
test_set = MyDataset(UCI_test.gesture_IMU, UCI_test.label, augment=False, time_warp=False, augmentor=augmentor)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)
# train_loader = DataLoader(shrec_train, batch_size=BATCH_SIZE, shuffle=True)
# test_loader = DataLoader(shrec_test, batch_size=BATCH_SIZE, shuffle=True)
# 使用 DataLoader 读取批次数据
train_iter = iter(train_loader)
data, label = next(train_iter)
# data = data.to(device)
# label = label.to(device)
# print(f'Data is on GPU: {data.is_cuda}')
# print(f'Labels is on GPU: {label.is_cuda}')
input_shape = data.shape
output_shape = UCI_train.gesture_num


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
    model_name = model_name,
    module=model,
    criterion=criterion,
    device=device,
    strategy=None,
    log_every_n_steps=10,
    gradient_accumulation_steps=1,
    detect_anomaly=True,
    clip_grad_norm=1.0,
    gesture_names=UCI_HAR_NAME
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