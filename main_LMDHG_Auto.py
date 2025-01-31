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
from dataset_module import GestureClass, LMDHG_NAME, LMDHG_dataset
import argparse
from model_factory_module import prepare_model
from train_module import TrainUnit
from augmentation import DataAugmentor
import numpy as np
from AutoML_hyperparameter import get_best_hyperparameters

# Instantiate the training dataset
Batch = Tuple[torch.Tensor, torch.Tensor]

Batch = Tuple[torch.Tensor, torch.Tensor]
LMDHG_train = LMDHG_dataset(
    data_path='LMDHG/my_xz_view/train', 
    dataset_name='LMDHG_train', 
)
LMDHG_test = LMDHG_dataset(
    data_path='LMDHG/my_xz_view/test', 
    dataset_name='LMDHG_test', 
)

# 读取数据
BATCH_SIZE = 64  # 设置批次大小

from torch.utils.data import Dataset, dataloader
# 实例化 DataAugmentor
augmentor = DataAugmentor()


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
            joints = self.augmentor.augment_data(joints)  # 调用 augment_data 函数
        return joints

    def __getitem__(self, idx):
        d = self.data[idx]  # 转为 numpy 数组进行数据增强        
        
        return d, self.labels[idx]
# 使用 augmentor 进行数据增强

# seed the RNG for better reproducibility. see docs https://pytorch.org/docs/stable/notes/randomness.html
seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('----------------------------\ndevice type:', device)
print("LMDHG_train.gesture_data.shape:", LMDHG_train.gesture_data.shape)
###dataset instance
train_set = MyDataset(LMDHG_train.gesture_data, LMDHG_train.label, augment=False, time_warp=False, augmentor=augmentor)
test_set = MyDataset(LMDHG_test.gesture_data, LMDHG_test.label, augment=False, time_warp=False, augmentor=augmentor)

print("train_set:", train_set)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)
# 使用 DataLoader 读取批次数据
train_iter = iter(train_loader)
data, label = next(train_iter)
print("data shape: ", data.shape)
input_shape = data.shape
output_shape = LMDHG_test.gesture_num

print('input shape:', input_shape)
print('output shape:', output_shape)

model_name = 'quartznet'

# 获取最佳超参数配置
best_config = get_best_hyperparameters(
    train_loader=train_loader,
    test_loader=test_loader,
    model_name=model_name,
    dataset_name='LMDHG',
    force_optimize=False  # 设置为True强制重新优化
)

# 使用最佳配置更新模型参数
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

path = f"./{model_name}_output/"
tb_logger = SummaryWriter(log_dir=path)

# 使用优化后的参数创建模型
model = prepare_model(model_name, input_shape, output_shape, device, **model_params)

criterion = nn.CrossEntropyLoss()
# 更新优化器参数
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=best_config["learning_rate"],
    weight_decay=best_config["weight_decay"]
)

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
    gesture_names=LMDHG_NAME
)

progress_bar = TQDMProgressBar(refresh_rate=10)
garbage_collector = GarbageCollector(step_interval=51)
callbacks = [
    garbage_collector,   
]

fit(
    my_unit,
    train_dataloader=train_loader,
    eval_dataloader=test_loader,
    max_epochs=400,
    callbacks=callbacks,
)

print("finished")