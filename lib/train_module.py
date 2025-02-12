import torch
import torch.nn as nn
from typing import Literal, Optional, Tuple, Union
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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dataset_module import GestureClass, Shrec2021Dataset, UCI_HAR_dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import argparse
import logging
import os
import tempfile
import uuid
from torch.utils.data.dataset import Dataset, TensorDataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.distributed import launcher as pet
from typing import List
from sklearn.metrics import classification_report
import json
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


Batch = Tuple[torch.Tensor, torch.Tensor]

class TrainUnit(AutoUnit[Batch]):
    def __init__(
        self,
        *,
        train_accuracy: MulticlassAccuracy,  # Metric for training accuracy
        eval_accuracy: MulticlassAccuracy,   # Metric for evaluation accuracy
        train_cm: MulticlassConfusionMatrix, # Metric for training confusion matrix
        eval_cm: MulticlassConfusionMatrix,  # Metric for evaluation confusion matrix
        log_every_n_steps: int,              # Log frequency for training steps
        tb_logger: Optional[TensorBoardLogger] = None,  # TensorBoard logger for monitoring
        model_name: str = 'quartznet',        # Default model name set to 'quartznet', best performance
        module: torch.nn.Module,             # Neural network model
        criterion: torch.nn.Module,          # Loss function for training
        device: Optional[torch.device] = None, # Device to run the model (CPU or GPU)
        strategy: Optional[Union[Strategy]] = None,  # Optional strategy (e.g., data parallelism)
        step_lr_interval: Literal["step", "epoch"] = "epoch",  # Learning rate step interval (per step or per epoch)
        precision: Optional[Union[str, torch.dtype]] = None,  # Precision (e.g., FP16)
        gradient_accumulation_steps: int = 1,  # Steps for gradient accumulation
        detect_anomaly: Optional[bool] = None, # Detect anomalies during training
        clip_grad_norm: Optional[float] = None, # Gradient norm clipping to prevent explosion
        clip_grad_value: Optional[float] = None, # Gradient value clipping
        swa_params: Optional[SWAParams] = None,  # Stochastic Weight Averaging parameters
        torch_compile_params: Optional[TorchCompileParams] = None,  # Torch compile parameters
        activation_checkpoint_params: Optional[ActivationCheckpointParams] = None,  # Activation checkpointing parameters
        training: bool = True,                 # Boolean flag to indicate training mode
        gesture_names: List[str],              # List of gesture class names
    ) -> None:
        super().__init__(
            module=module,
            device=device,
            strategy=strategy,
            step_lr_interval=step_lr_interval,
            precision=precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            detect_anomaly=detect_anomaly,
            clip_grad_norm=clip_grad_norm,
            clip_grad_value=clip_grad_value,
            swa_params=swa_params,
            torch_compile_params=torch_compile_params,
            activation_checkpoint_params=activation_checkpoint_params,
            training=training,
        )
        self.criterion = criterion
        self.tb_logger = tb_logger
        self.train_accuracy = train_accuracy
        self.eval_accuracy = eval_accuracy
        self.train_cm = train_cm
        self.eval_cm = eval_cm
        self.log_every_n_steps = log_every_n_steps
        self.gesture_names = gesture_names
        self.model_name = model_name
        self.best_eval_accuracy = 0.0

    def configure_optimizers_and_lr_scheduler(
        self, module: torch.nn.Module
    ) -> Tuple[torch.optim.Optimizer, Optional[TLRScheduler]]:
        """
        Configure the optimizer and learning rate scheduler for training.

        Args:
            module (torch.nn.Module): The model to optimize.
        
        Returns:
            Tuple[torch.optim.Optimizer, Optional[TLRScheduler]]: The optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.Adam(module.parameters(), lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
        return optimizer, lr_scheduler

    def compute_loss(
        self, state: State, data: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the loss between predicted and true labels.

        Args:
            state (State): Current training state.
            data (Batch): Input batch consisting of features and targets.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The loss value and the model's output.
        """
        inputs, targets = data
        inputs = inputs.to(self.device).float()
        targets = targets.to(self.device).long()
        outputs = self.module(inputs).float()
        loss = self.criterion(outputs, targets)
        return loss, outputs

    def on_train_step_end(
        self,
        state: State,
        data: Batch,
        step: int,
        results: TrainStepResults,
    ) -> None:
        """
        Actions to take after each training step.

        Args:
            state (State): Current training state.
            data (Batch): Input batch consisting of features and targets.
            step (int): The current step number.
            results (TrainStepResults): The results of the current step including loss and outputs.
        """
        loss, outputs = results.loss.float(), results.outputs.float()
        _, targets = data
        targets = targets.to(self.device).long()

        # Update training metrics
        self.train_accuracy.update(outputs, targets)
        self.train_cm.update(outputs, targets)

        # Log training metrics to TensorBoard every specified number of steps
        if step % self.log_every_n_steps == 0 and self.tb_logger is not None:
            accuracy = self.train_accuracy.compute()
            self.tb_logger.add_scalar("train_accuracy", accuracy, step)
            self.tb_logger.add_scalar("train_loss", loss, step)

    def on_train_epoch_end(self, state: State) -> None:
        """
        Actions to take after each training epoch ends.

        Args:
            state (State): Current training state.
        """
        super().on_train_epoch_end(state)
        cm = self.train_cm.compute()
        img = self.plot_cm(cm, self.gesture_names)
        epoch = self.train_progress.num_epochs_completed
        if self.tb_logger:
            self.tb_logger.add_figure("train_cm", img, epoch)

        # Save model checkpoint every 10 epochs
        if epoch % 10 == 0:
            output_dir = f"./{self.model_name}_output/checkpoints/"
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            torch.save(self.module.state_dict(), f"{output_dir}/checkpoint_{epoch}.pt")

        self.train_accuracy.reset()
        self.train_cm.reset()

    def on_eval_step_end(
        self,
        state: State,
        data: Batch,
        step: int,
        loss: torch.Tensor,
        outputs: torch.Tensor,
    ) -> None:
        """
        Actions to take after each evaluation step.

        Args:
            state (State): Current evaluation state.
            data (Batch): Input batch consisting of features and targets.
            step (int): The current step number.
            loss (torch.Tensor): Loss value at this step.
            outputs (torch.Tensor): Model outputs.
        """
        _, targets = data
        targets = targets.to(self.device).long()

        # Update evaluation metrics
        self.eval_accuracy.update(outputs, targets)
        self.eval_cm.update(outputs, targets)

    def on_eval_end(self, state: State) -> None:
        """
        Actions to take after evaluation ends.

        Args:
            state (State): Current evaluation state.
        """
        accuracy = self.eval_accuracy.compute()
        cm = self.eval_cm.compute()
        img = self.plot_cm(cm, self.gesture_names)

        if self.tb_logger is not None:
            self.tb_logger.add_scalar("eval_accuracy", accuracy, self.eval_progress.num_steps_completed)
            self.tb_logger.add_figure("eval_cm", img, self.eval_progress.num_epochs_completed)

        self.eval_accuracy.reset()
        self.eval_cm.reset()

        # Generate classification report from confusion matrix
        cm_np = cm.cpu().numpy()
        all_targets = []
        all_preds = []
        for i in range(cm_np.shape[0]):
            for j in range(cm_np.shape[1]):
                count = int(cm_np[i, j])
                all_targets.extend([i] * count)
                all_preds.extend([j] * count)

        # Generate and print classification report
        report = classification_report(all_targets, all_preds, target_names=self.gesture_names, digits=4)

        # Save best model and evaluation report if improved
        if accuracy > self.best_eval_accuracy:
            self.best_eval_accuracy = accuracy
            self.best_eval_report = report
            self.best_eval_epoch = self.eval_progress.num_epochs_completed

            # Save the best model
            best_model_path = f"./{self.model_name}_output/best_model.pth"
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(self.module.state_dict(), best_model_path)
            print(f"Best model parameters saved to: {best_model_path}")
            print(f"Best eval_accuracy so far: {self.best_eval_accuracy} at epoch {self.best_eval_epoch}")
            print(self.best_eval_report)

            # Save best evaluation report and confusion matrix image
            save_path = f"./{self.model_name}_output/best_eval_report.txt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                f.write(f"Best Eval Accuracy: {self.best_eval_accuracy:.4f}\n")
                f.write(f"Best Eval Epoch: {self.best_eval_epoch}\n")
                f.write(f"Best Eval Report:\n{self.best_eval_report}\n")

            confusion_matrix_path = f"./{self.model_name}_output/best_confusion_matrix.pdf"
            img.savefig(confusion_matrix_path, format='pdf')
            print(f"Best confusion matrix saved to: {confusion_matrix_path}")
            plt.close(img.figure)

    def plot_cm(self, cm_metric, gesture_names, fig_size=(10, 7), cmap="Spectral", normalize=True):
        """
        Plot the confusion matrix.

        Args:
            cm_metric: Confusion matrix metric to visualize.
            gesture_names (List[str]): List of class names.
            fig_size (tuple): Figure size for the plot.
            cmap (str): Colormap for the heatmap.
            normalize (bool): Whether to normalize the confusion matrix.
        
        Returns:
            plt.Figure: The resulting confusion matrix plot.
        """
        cm = cm_metric.cpu().numpy()
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        df_cm = pd.DataFrame(cm, index=gesture_names, columns=gesture_names)
        plt.figure(figsize=fig_size)
        fig = sns.heatmap(df_cm, annot=True, cmap=cmap, fmt=".2f").get_figure()

        return fig