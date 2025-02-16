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
from AutoMR_hyperparameter import get_best_hyperparameters
import numpy as np

# Define the data type for batches
Batch = Tuple[torch.Tensor, torch.Tensor]

# Load the training and testing datasets
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
    annotation_path='UCI_HAR/y_train.txt'  # Path to training labels
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
    annotation_path='UCI_HAR/y_test.txt'  # Path to testing labels
)

# Custom dataset class to handle data augmentation and reshaping
class MyDataset(Dataset):
    def __init__(self, data, labels, augment=False, time_warp=False, augmentor=None):
        self.data = data
        self.labels = labels
        self.augment = augment  # Whether to apply data augmentation
        self.time_warp = time_warp  # Whether to apply time warping
        self.augmentor = augmentor  # The augmentor object for data augmentation

    def __len__(self):
        return len(self.data)  # Return the size of the dataset

    # Function to apply data augmentation to the sequence
    def augment_sequence(self, joints):
        if self.augmentor is not None:
            joints = self.augmentor.augment_data(joints)  # Apply augmentation
        return joints

    # Fetch data for a given index and apply transformations
    def __getitem__(self, idx):
        d = self.data[idx]  # Get the data point
        if self.augment:  # If augmentation is enabled, apply it
            d = self.augment_sequence(d)
        if self.time_warp:  # If time warping is enabled, apply it
            d = self.augmentor.time_warp(d)
        d = np.reshape(d, (33, 17))  # Reshape the data to the desired dimensions
        return torch.from_numpy(d), self.labels[idx]  # Return the data as a tensor along with the label

# Main function to setup and train the model
def main():
    # Set random seed for reproducibility
    seed(0)
    
    # Set the device to GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Data preparation
    BATCH_SIZE = 32
    augmentor = DataAugmentor()  # Initialize the data augmentor
    
    # Create the training and testing datasets and data loaders
    train_set = MyDataset(UCI_train.gesture_IMU, UCI_train.label, 
                         augment=False, time_warp=False, augmentor=augmentor)
    test_set = MyDataset(UCI_test.gesture_IMU, UCI_test.label, 
                        augment=False, time_warp=False, augmentor=augmentor)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    # Get the input and output shapes by inspecting a batch of data
    data, _ = next(iter(train_loader))
    input_shape = data.shape
    output_shape = UCI_train.gesture_num  # Number of output classes

    # Model configuration
    model_name = 'quartznet'
    
    # Retrieve the best hyperparameters through AutoML
    best_config = get_best_hyperparameters(train_loader, test_loader, model_name)
    
    # Set up the model and TensorBoard logger
    path = f"./{model_name}_output/"
    tb_logger = SummaryWriter(log_dir=path)
    model = prepare_model(model_name, input_shape, output_shape, device)  # Prepare the model based on the name

    # Set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_config["learning_rate"],  # Use the learning rate from the best configuration
        weight_decay=best_config["weight_decay"]  # Use the weight decay from the best configuration
    )
    
    # Initialize evaluation metrics
    train_accuracy = MulticlassAccuracy(device=device)
    eval_accuracy = MulticlassAccuracy(device=device)
    train_cm = MulticlassConfusionMatrix(num_classes=output_shape, device=device)
    eval_cm = MulticlassConfusionMatrix(num_classes=output_shape, device=device)

    # Create the training unit
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
        log_every_n_steps=10,  # Log every 10 steps
        gradient_accumulation_steps=1,
        detect_anomaly=True,  # Enable anomaly detection
        clip_grad_norm=1.0,  # Clip gradients to prevent explosion
        gesture_names=UCI_HAR_NAME  # Name of the gestures
    )

    # Set up the callbacks
    callbacks = [GarbageCollector(step_interval=51)]  # Garbage collection to clean up memory

    # Start the training process
    fit(
        my_unit,
        train_dataloader=train_loader,
        eval_dataloader=test_loader,
        max_epochs=400,  # Run for 400 epochs
        callbacks=callbacks,
    )

# Run the main function
if __name__ == "__main__":
    main()
