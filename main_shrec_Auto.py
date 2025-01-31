import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import MulticlassAccuracy, MulticlassConfusionMatrix
from torchtnt.framework.fit import fit
from torchtnt.framework.callbacks.garbage_collector import GarbageCollector
from torchtnt.framework.callbacks.tqdm_progress_bar import TQDMProgressBar
from dataset_module import Shrec2021Dataset, Modality, SHREC_NAME
from model_factory_module import prepare_model
from train_module import TrainUnit
from AutoML_hyperparameter import get_best_hyperparameters
from torchtnt.utils import init_from_env, seed
import os
from augmentation import DataAugmentor

# Seed the RNG for better reproducibility
seed(0)

# Initialize the device
device = init_from_env()

# Hyperparameters
BATCH_SIZE = 64  # Default batch size
EPOCHS = 400  # Default number of epochs

# Load the SHREC2021 dataset
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
    annotation_path='shrec/annotation_shrec2021_train.txt'  # 提供注释文件路径
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
    annotation_path='shrec/annotation_shrec2021_test.txt'  # 提供注释文件路径
)
# 读取shrec2021数据
shrec_modality = shrec_train.read_dataset(shrec_train.data_path)

# 打印可用的模态以确认键值
print('Available modalities in the training dataset:', shrec_modality.keys())

# 获取相应的注释和标签
# train_joint_data, train_label = shrec_train.gesture_data, shrec_train.labels
BATCH_SIZE = 128  # 设置批次大小

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
        return torch.tensor(d), self.labels[idx]
# 使用 augmentor 进行数据增强

# seed the RNG for better reproducibility. see docs https://pytorch.org/docs/stable/notes/randomness.html
seed(0)

# device and process group initialization, gpu incompaitible:
device = init_from_env()
# device = torch.device('cpu')
print('device type---------------------------------', device)


###1. Shrec2021 dataset instance
train_set = MyDataset(shrec_train.gesture_data, shrec_train.label, augment=True, time_warp=True, augmentor=augmentor)
test_set = MyDataset(shrec_test.gesture_data, shrec_test.label, augment=False, time_warp=False, augmentor=augmentor)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

# Get best hyperparameters using AutoML
print("Optimizing hyperparameters using AutoML...")
best_config = get_best_hyperparameters(
    train_loader=train_loader,
    test_loader=test_loader,
    model_name="quartznet",
    dataset_name="SHREC",
    force_optimize=False  # Change to True if you want to re-optimize
)
print("Best hyperparameters found:", best_config)

# Prepare the model
input_shape = next(iter(train_loader))[0].shape
output_shape = len(SHREC_NAME)
model = prepare_model(
    model_type="quartznet",
    input_shape=input_shape,
    output_shape=output_shape,
    device=device,
    **best_config
)

# Training setup
tb_logger = SummaryWriter(log_dir="./quartznet_output/")
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
    model_name='quartznet',
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

# Callbacks
progress_bar = TQDMProgressBar(refresh_rate=10)
garbage_collector = GarbageCollector(step_interval=51)

# Train the model
print("Starting training...")
fit(
    my_unit,
    train_dataloader=train_loader,
    eval_dataloader=test_loader,
    max_epochs=EPOCHS,
    callbacks=[progress_bar, garbage_collector]
)

# Save the model
model_save_path = "./quartznet_output/quartznet_best_model.pth"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")