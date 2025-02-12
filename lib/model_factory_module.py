import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy, MulticlassConfusionMatrix
from torchtnt.framework.auto_unit import AutoUnit, TrainStepResults
from torchtnt.framework.fit import fit
from torchtnt.framework.state import State
from torchtnt.framework.callbacks.tqdm_progress_bar import TQDMProgressBar
from torchtnt.framework.callbacks.garbage_collector import GarbageCollector
from torchtnt.utils.loggers import TensorBoardLogger
from dataset_module import GestureClass
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



class QuartzNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(QuartzNetBlock, self).__init__()
        self.depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride, padding=padding,
                                        dilation=dilation, groups=in_channels, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.batch_norm(x)
        x += residual  # Residual connection
        x = self.relu(x)
        return x

class QuartzNet(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(QuartzNet, self).__init__()

        batch_size, seq_length, num_channels = input_shape
        num_classes = output_shape
        # Initial convolution
        self.conv = nn.Conv1d(in_channels=num_channels, out_channels=256, kernel_size=33, stride=2, padding=16, bias=False)
        self.bn = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        # QuartzNet blocks
        self.block1 = self._make_block(256, 256, kernel_size=33, stride=1, padding=16, dilation=1)
        self.block2 = self._make_block(256, 256, kernel_size=39, stride=1, padding=19, dilation=1)
        self.block3 = self._make_block(256, 256, kernel_size=51, stride=1, padding=25, dilation=1)
        self.block4 = self._make_block(256, 256, kernel_size=63, stride=1, padding=31, dilation=1)
        self.block5 = self._make_block(256, 256, kernel_size=75, stride=1, padding=37, dilation=1)


        # Final layers
        self.dropout = nn.Dropout(p=0.25)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_block(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        return nn.Sequential(
            QuartzNetBlock(in_channels, out_channels, kernel_size, stride, padding, dilation),
            QuartzNetBlock(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.dropout(x)
        x = self.avg_pool(x).squeeze(-1)
        x = self.fc(x)

        return x
    


class BasicBlock(nn.Module):
    def __init__(self, num_cells=5, kernel_size=33, in_channels=256,
                 out_channels=256, dropout_rate=0.25,
                 norm_layer=None, activation=None):
        super(BasicBlock, self).__init__()

        self.num_cells = num_cells
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # padding = 'same' for stride = 1, dilation = 1
        self.padding = (self.kernel_size - 1) // 2

        self.norm_layer = nn.BatchNorm1d if norm_layer is None else norm_layer
        self.activation = nn.ReLU if activation is None else activation

        def build_cell(in_channels, out_channels):
            return nn.Sequential(
                # 1D Depthwise Convolution
                nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                          kernel_size=self.kernel_size, padding=self.padding,
                          groups=in_channels),
                # Pointwise Convolution
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=1),
                # Normalization
                self.norm_layer(num_features=out_channels),
            )

        self.cells = [build_cell(self.in_channels, self.out_channels)]
        for i in range(1, self.num_cells):
            self.cells.append(self.activation())
            self.cells.append(nn.Dropout(p=dropout_rate))
            self.cells.append(build_cell(self.out_channels, self.out_channels))
        self.cells = nn.Sequential(*self.cells)

        # Skip connection
        self.residual = nn.Sequential(
            # Pointwise Convolution
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1),
            # Normalization
            self.norm_layer(num_features=out_channels),
        )
        self.activation = self.activation()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = inputs
        outputs = self.cells(outputs)
        outputs = self.activation(outputs + self.residual(inputs))
        outputs = self.dropout(outputs)
        return outputs


class ResidualModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.adjust_channels = None

        if in_channels != out_channels:
            self.adjust_channels = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        residual = x
        if self.adjust_channels:
            residual = self.adjust_channels(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x += residual
        x = self.bn(x)
        x = self.relu(x)

        return x


class QuartzNetLarge(nn.Module):
    def __init__(self, input_shape, output_shape, num_blocks=5, num_cells=5,
                 input_kernel=33, input_channels=256,
                 head_kernel=87, head_channels=512,
                 block_kernels=(33, 39, 51, 63, 75),
                 block_channels=(256, 256, 256, 512, 512),
                 dropout_rate=0.25, norm_layer=None, activation=None):
        super(QuartzNet, self).__init__()

        batch_size, seq_length, num_channels = input_shape

        self.norm_layer = nn.BatchNorm1d if norm_layer is None else norm_layer
        self.activation = nn.ReLU if activation is None else activation
        self.num_blocks = num_blocks
        self.num_cells = num_cells
        self.num_mels = num_channels
        self.num_labels = output_shape

        # padding to reduce time frames (T) -> (T / 2)
        input_padding = (input_kernel - 1) // 2
        self.input = nn.Sequential(
            # C1 Block: Conv-BN-ReLU
            nn.Conv1d(in_channels=self.num_mels, out_channels=self.num_mels,
                      kernel_size=input_kernel, stride=2,
                      padding=input_padding, groups=self.num_mels),
            nn.Conv1d(in_channels=self.num_mels, out_channels=input_channels,
                      kernel_size=1),
            self.norm_layer(num_features=input_channels),
            self.activation(),
            nn.Dropout(p=dropout_rate)
        )

        in_channels = input_channels
        self.blocks = []
        self.residuals = []
        for i in range(self.num_blocks):
            self.blocks.append(
                BasicBlock(num_cells=self.num_cells, kernel_size=block_kernels[i],
                           in_channels=in_channels, out_channels=block_channels[i],
                           dropout_rate=dropout_rate, norm_layer=self.norm_layer, activation=self.activation)
            )
            self.residuals.append(
                nn.Conv1d(in_channels=in_channels, out_channels=block_channels[i], kernel_size=1)
            )
            in_channels = block_channels[i]
        self.blocks = nn.ModuleList(self.blocks)
        self.residuals = nn.ModuleList(self.residuals)

        # padding = 'same' for stride = 1, dilation = 2
        head_padding = head_kernel - 1
        self.head = nn.Sequential(
            # C2 Block: Conv-BN-ReLU
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=head_kernel, dilation=2,
                      padding=head_padding, groups=in_channels),
            nn.Conv1d(in_channels=in_channels, out_channels=head_channels,
                      kernel_size=1),
            self.norm_layer(num_features=head_channels),
            self.activation(),
            nn.Dropout(p=dropout_rate),
            # C3 Block: Conv-BN-ReLU
            nn.Conv1d(in_channels=head_channels, out_channels=2 * head_channels,
                      kernel_size=1),
            self.norm_layer(num_features=2 * head_channels),
            self.activation(),
            nn.Dropout(p=dropout_rate),
            # C4 Block: Pointwise Convolution
            nn.Conv1d(in_channels=2 * head_channels, out_channels=self.num_labels,
                      kernel_size=1)
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.input(inputs)
        for block, residual in zip(self.blocks, self.residuals):
            outputs = block(outputs) + residual(outputs)
        outputs = self.head(outputs)
        outputs = self.avg_pool(outputs).squeeze(-1)
        return outputs


def make_quartznet(input_shape, output_shape, params):
    return QuartzNetLarge(input_shape, output_shape,
                    #  num_blocks=params['num_blocks'], num_cells=params['num_cells'],
                    #  input_kernel=params['input_kernel'], input_channels=params['input_channels'],
                    #  head_kernel=params['head_kernel'], head_channels=params['head_channels'],
                    #  block_kernels=params['block_kernels'], block_channels=params['block_channels'],
                     dropout_rate=params['dropout_rate'])

# ... existing imports ...

def prepare_model(model_type, input_shape, output_shape, device: torch.device, **kwargs) -> nn.Module:
    """
    Instantiate model
    Args:
        model_type (str): The type of the model (e.g., "quartznet", "resnet", etc.)
        input_shape (tuple): The shape of the input data (e.g., (channels, height, width))
        output_shape (int): The number of output classes or dimensions
        device (torch.device): The device on which the model will run (e.g., CPU or GPU)
        **kwargs: Additional model-specific parameters (e.g., layers, activation functions, etc.)
    """

    if model_type == "quartznet":
        # 如果有额外参数，更新模型的dropout
        model = QuartzNet(input_shape, output_shape)
        if 'dropout_rate' in kwargs:
            model.dropout.p = kwargs['dropout_rate']
    elif model_type == "quartznet_large":
        params = {"dropout_rate": kwargs.get('dropout_rate', 0.3)}
        model = make_quartznet(input_shape, output_shape, params)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    # print(f'here deploying the selected model to be: {model_type}!')
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    # print('device type:', device)
    return model.to(device)




