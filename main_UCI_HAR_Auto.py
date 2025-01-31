from typing import Tuple
import torch
import torch.nn as nn
from dataset_module import Modality, UCI_HAR_NAME
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import MulticlassAccuracy, MulticlassConfusionMatrix
from torchtnt.utils import seed
from torchtnt.framework.fit import fit
from torchtnt.framework.callbacks.garbage_collector import GarbageCollector

from dataset_module import UCI_HAR_dataset
from model_factory_module import prepare_model
from train_module import TrainUnit
from augmentation import DataAugmentor
from AutoML_hyperparameter import get_best_hyperparameters
import numpy as np

# 定义数据类型
Batch = Tuple[torch.Tensor, torch.Tensor]

# 加载训练和测试数据集
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
    annotation_path='UCI_HAR/y_train.txt'
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
    annotation_path='UCI_HAR/y_test.txt'
)

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
        d = np.reshape(d, (33, 17))
        return torch.from_numpy(d), self.labels[idx]

def main():
    # 设置随机种子
    seed(0)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # 数据准备
    BATCH_SIZE = 32
    augmentor = DataAugmentor()
    
    # 创建数据集和数据加载器
    train_set = MyDataset(UCI_train.gesture_IMU, UCI_train.label, 
                         augment=False, time_warp=False, augmentor=augmentor)
    test_set = MyDataset(UCI_test.gesture_IMU, UCI_test.label, 
                        augment=False, time_warp=False, augmentor=augmentor)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    # 获取输入形状和输出形状
    data, _ = next(iter(train_loader))
    input_shape = data.shape
    output_shape = UCI_train.gesture_num

    # 模型配置
    model_name = 'quartznet'
    
    # 获取最佳超参数
    best_config = get_best_hyperparameters(train_loader, test_loader, model_name)
    
    # 创建模型和日志记录器
    path = f"./{model_name}_output/"
    tb_logger = SummaryWriter(log_dir=path)
    model = prepare_model(model_name, input_shape, output_shape, device)

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
        gesture_names=UCI_HAR_NAME
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

if __name__ == "__main__":
    main()