import os
from typing import Literal, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import MulticlassAccuracy, MulticlassConfusionMatrix
from torchtnt.framework.fit import fit
from torchtnt.framework.callbacks.garbage_collector import GarbageCollector
from torchtnt.framework.callbacks.tqdm_progress_bar import TQDMProgressBar
from torchtnt.utils import seed
from dataset_module import Modality, OPPO_NAME, mhealth_dataset
from model_factory_module import prepare_model
from train_module import TrainUnit
from augmentation import DataAugmentor
from AutoML_hyperparameter import get_best_hyperparameters

# 定义Batch类型
Batch = Tuple[torch.Tensor, torch.Tensor]

# 数据集实例化
oppo_train = mhealth_dataset(
    data_path='oppo/clean_data/processed/s4X_train.npy', 
    dataset_name='oppo_train',  
    data_modality=[Modality.IMU], 
    timestamps=None,
    split_joint_name=None,
    sensor_name='imu',
    sampling_rate=30, 
    padding_value=0,
    window_size=15,
    stride=7,
    annotation_path='oppo/clean_data/processed/s4y_train.txt'
)

oppo_test = mhealth_dataset(
    data_path='oppo/clean_data/processed/s4X_test.npy', 
    dataset_name='oppo_test',  
    data_modality=[Modality.IMU], 
    timestamps=None,
    split_joint_name=None,
    sensor_name='imu',
    sampling_rate=30, 
    padding_value=0,
    window_size=15,
    stride=7,
    annotation_path='oppo/clean_data/processed/s4y_test.txt'
)

# 数据增强器
augmentor = DataAugmentor()

# 定义数据集类
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

# 标签映射函数
def remap_labels(labels):
    unique_labels = sorted(np.unique(labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    remapped_labels = np.array([label_map[label] for label in labels])
    return remapped_labels

# 主函数
def main():
    seed(0)  # 设置随机种子
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # 加载数据
    X_train = oppo_train.gesture_IMU
    y_train = remap_labels(oppo_train.label)
    X_test = oppo_test.gesture_IMU
    y_test = remap_labels(oppo_test.label)

    # 数据加载器
    BATCH_SIZE = 256
    train_set = MyDataset(X_train, y_train, augment=False, time_warp=False, augmentor=augmentor)
    test_set = MyDataset(X_test, y_test, augment=False, time_warp=False, augmentor=augmentor)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    # 输入和输出形状
    input_shape = next(iter(train_loader))[0].shape
    output_shape = len(OPPO_NAME)
    print('Input shape:', input_shape)
    print('Output shape:', output_shape)

    # 模型配置
    model_name = 'quartznet'
    best_config = get_best_hyperparameters(train_loader, test_loader, model_name, "OPPO")  # 获取最佳超参数

    # 准备模型
    path = f"./{model_name}_output_auto/"
    tb_logger = SummaryWriter(log_dir=path)
    model = prepare_model(model_name, input_shape, output_shape, device)

    # 设置优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_config["learning_rate"],
        weight_decay=best_config["weight_decay"]
    )

    # 评估指标
    train_accuracy = MulticlassAccuracy(device=device)
    eval_accuracy = MulticlassAccuracy(device=device)
    train_cm = MulticlassConfusionMatrix(num_classes=output_shape, device=device)
    eval_cm = MulticlassConfusionMatrix(num_classes=output_shape, device=device)

    # 训练单元
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
        gesture_names=OPPO_NAME
    )

    # 回调函数
    callbacks = [
        TQDMProgressBar(refresh_rate=10),
        GarbageCollector(step_interval=51),
    ]

    # 开始训练
    fit(
        my_unit,
        train_dataloader=train_loader,
        eval_dataloader=test_loader,
        max_epochs=400,
        callbacks=callbacks,
    )

    print("Training complete")

if __name__ == "__main__":
    main()