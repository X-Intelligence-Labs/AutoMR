from typing import Tuple
import torch
import torch.nn as nn
from dataset_module import Modality, DB4_NAME
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import MulticlassAccuracy, MulticlassConfusionMatrix
from torchtnt.utils import seed
from torchtnt.framework.fit import fit
from torchtnt.framework.callbacks.garbage_collector import GarbageCollector

from dataset_module import DB4_dataset
from model_factory_module import prepare_model
from train_module import TrainUnit
from augmentation import DataAugmentor
from AutoML_hyperparameter import get_best_hyperparameters
import numpy as np

# 定义数据类型
Batch = Tuple[torch.Tensor, torch.Tensor]

# 加载训练和测试数据集
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
    annotation_path='db4/y_train.txt'
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
    annotation_path='db4/y_test.txt'
)

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
            joints = self.augmentor.augment_data(joints)
        return joints

    def __getitem__(self, idx):
        d = self.data[idx]
        if self.augment:
            d = self.augment_sequence(d)
        if self.time_warp:
            d = self.augmentor.time_warp(d)
        d = np.reshape(d, (520, 12))
        return torch.from_numpy(d), self.labels[idx]

def sample_data(X, y, label, ratio):
    """随机采样指定比例的数据，避免类不平衡问题"""
    target_indices = np.where(y == label)[0]
    other_indices = np.where(y != label)[0]
    sample_size = int(len(target_indices) * ratio)
    sampled_indices = np.random.choice(target_indices, sample_size, replace=False)
    final_indices = np.concatenate([sampled_indices, other_indices])
    np.random.shuffle(final_indices)
    return X[final_indices], y[final_indices]

def main():
    # 设置随机种子
    seed(0)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据准备
    BATCH_SIZE = 64  # 初始batch_size，不影响超参数优化
    augmentor = DataAugmentor()
    
    # 加载和平衡数据
    X_train = db4_train.gesture_EMG
    y_train = db4_train.label
    X_test = db4_test.gesture_EMG
    y_test = db4_test.label
    
    # 数据平衡处理
    X_train_balance, y_train_balance = sample_data(X_train, y_train, label=0, ratio=0.02)
    X_test_balance, y_test_balance = sample_data(X_test, y_test, label=0, ratio=0.02)
    
    # 创建数据集和加载器
    train_set = MyDataset(X_train_balance, y_train_balance, augment=False, time_warp=False, augmentor=augmentor)
    test_set = MyDataset(X_test_balance, y_test_balance, augment=False, time_warp=False, augmentor=augmentor)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)
    
    # 获取输入形状
    data, _ = next(iter(train_loader))
    input_shape = data.shape
    output_shape = db4_train.gesture_num
    
    # 模型配置
    model_name = 'quartznet'
    
    # 获取最佳超参数
    best_config = get_best_hyperparameters(
        train_loader=train_loader,
        test_loader=test_loader,
        model_name=model_name,
        dataset_name="DB4",
        force_optimize=True  # 如果已存在最佳配置文件，可设置为False
    )
    
    # 使用最佳超参数更新数据加载器
    train_loader = DataLoader(
        train_set,
        batch_size=int(best_config["batch_size"]),
        shuffle=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=int(best_config["batch_size"]),
        shuffle=False
    )
    
    # 创建模型和日志记录器
    path = f"./{model_name}_output/"
    tb_logger = SummaryWriter(log_dir=path)
    model_params = {
        "dropout_rate": best_config["dropout_rate"],
        "num_blocks": int(best_config["num_blocks"]),
        "num_cells": int(best_config["num_cells"]),
        "input_channels": int(best_config["input_channels"]),
        "head_channels": int(best_config["head_channels"]),
        "input_kernel": int(best_config["input_kernel"]),
        "head_kernel": int(best_config["head_kernel"])
    }
    model = prepare_model(model_name, input_shape, output_shape, device, **model_params)
    
    # 设置训练组件
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_config["learning_rate"],
        weight_decay=best_config["weight_decay"]
    )
    
    # 设置评估指标
    train_accuracy = MulticlassAccuracy(device=device)
    eval_accuracy = MulticlassAccuracy(device=device)
    train_cm = MulticlassConfusionMatrix(num_classes=output_shape, device=device)
    eval_cm = MulticlassConfusionMatrix(num_classes=output_shape, device=device)
    
    # 创建训练单元
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
    
    # 设置回调函数
    callbacks = [GarbageCollector(step_interval=51)]
    
    # 开始训练
    fit(
        my_unit,
        train_dataloader=train_loader,
        eval_dataloader=test_loader,
        max_epochs=400,
        callbacks=callbacks,
    )
    
    print("Training finished")

if __name__ == "__main__":
    main()