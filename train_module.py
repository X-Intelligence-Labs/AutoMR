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
        train_accuracy: MulticlassAccuracy,
        eval_accuracy: MulticlassAccuracy,
        train_cm: MulticlassConfusionMatrix,
        eval_cm: MulticlassConfusionMatrix,
        log_every_n_steps: int,
        tb_logger: Optional[TensorBoardLogger] = None,
        model_name: str = 'quartznet', #default set as quartznet, best peformance.
        module: torch.nn.Module,
        criterion: torch.nn.Module,
        device: Optional[torch.device] = None,
        strategy: Optional[Union[Strategy]] = None,
        step_lr_interval: Literal["step", "epoch"] = "epoch",
        precision: Optional[Union[str, torch.dtype]] = None,
        gradient_accumulation_steps: int = 1,
        detect_anomaly: Optional[bool] = None,
        clip_grad_norm: Optional[float] = None,
        clip_grad_value: Optional[float] = None,
        swa_params: Optional[SWAParams] = None,
        torch_compile_params: Optional[TorchCompileParams] = None,
        activation_checkpoint_params: Optional[ActivationCheckpointParams] = None,
        training: bool = True,
        gesture_names: List[str],
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
        # create accuracy metrics to compute the accuracy of training and evaluation
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
    ) -> Tuple[torch.optim.Optimizer, Optional[TLRScheduler]]: # type: ignore
        optimizer = torch.optim.Adam(module.parameters(), lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
        return optimizer, lr_scheduler

    def compute_loss(
        self, state: State, data: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = data
        inputs = inputs.to(self.device)
        targets = targets.to(self.device).long()
        # print(f'Compute_loss, Train inputs is on GPU: {inputs.is_cuda}')
        # print(f'Compute_loss,Train targets is on GPU: {targets.is_cuda}')
        inputs = inputs.float()
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
        loss, outputs = results.loss.float(), results.outputs.float()
        _, targets = data
        targets = targets.long()
        targets = targets.to(self.device)
        # print(f'on_train_step_end, Train targets is on GPU: {targets.is_cuda}')
        self.train_accuracy.update(outputs, targets)
        self.train_cm.update(outputs, targets)

        if step % self.log_every_n_steps == 0 and self.tb_logger is not None:
            accuracy = self.train_accuracy.compute()
            self.tb_logger.add_scalar("train_accuracy", accuracy, step)
            self.tb_logger.add_scalar("train_loss", loss, step)

            # print(f"train_loss at step {step}: {loss}")

    def on_train_epoch_end(self, state: State) -> None:
        super().on_train_epoch_end(state)
        # reset the metric every epoch
        cm = self.train_cm.compute()
        img = self.plot_cm(cm, self.gesture_names)
        epoch = self.train_progress.num_epochs_completed
        if self.tb_logger:
            self.tb_logger.add_figure("train_cm", img, epoch)
        if epoch % 10 == 0:
            output_dir = f"./{self.model_name}_output/checkpoints/"
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            torch.save(self.module.state_dict(), f"{output_dir}/checkpoint_{epoch}.pt")

        # print(f"train_cm at epoch {epoch}: {cm}")
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
        _, targets = data
        targets = targets.to(self.device).long()
        # print(f'on_eval_step_end, Train targets is on GPU: {targets.is_cuda}')
        self.eval_accuracy.update(outputs, targets)
        self.eval_cm.update(outputs, targets)
        

    def on_eval_end(self, state: State) -> None:
        accuracy = self.eval_accuracy.compute()
        cm = self.eval_cm.compute()
        img = self.plot_cm(cm, self.gesture_names)

        if self.tb_logger is not None:
            self.tb_logger.add_scalar("eval_accuracy", accuracy, self.eval_progress.num_steps_completed)
            self.tb_logger.add_figure("eval_cm", img, self.eval_progress.num_epochs_completed)

        # print(f"eval_accuracy: {accuracy}")
        
        self.eval_accuracy.reset()
        self.eval_cm.reset()

        # 从混淆矩阵中提取真实标签和预测标签
        cm_np = cm.cpu().numpy()
        all_targets = []
        all_preds = []
        for i in range(cm_np.shape[0]):
            for j in range(cm_np.shape[1]):
                count = int(cm_np[i, j])
                all_targets.extend([i] * count)
                all_preds.extend([j] * count)

        # 生成并打印分类报告
        report = classification_report(all_targets, all_preds, target_names=self.gesture_names, digits=4)

    # 记录并打印最高eval acc
        if not hasattr(self, 'best_eval_accuracy') or accuracy > self.best_eval_accuracy:
            self.best_eval_accuracy = accuracy
            self.best_eval_report = report
            self.best_eval_epoch = self.eval_progress.num_epochs_completed

            # 保存模型参数
            best_model_path = f"./{self.model_name}_output/best_model.pth"
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(self.module.state_dict(), best_model_path)
            print(f"Best model parameters saved to: {best_model_path}")
            # 打印最高准确率和报告
            print(f"Best eval_accuracy so far: {self.best_eval_accuracy} at epoch {self.best_eval_epoch}")
            print(self.best_eval_report)
            
            save_path = f"./{self.model_name}_output/best_eval_report.txt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # 构造保存内容
            best_eval_data = (
                f"Best Eval Accuracy: {self.best_eval_accuracy:.4f}\n"
                f"Best Eval Epoch: {self.best_eval_epoch}\n"
                f"Best Eval Report:\n"
                f"{self.best_eval_report}\n"
            )

            # 将结果写入文件（覆盖模式）
            with open(save_path, "w") as f:
                f.write(best_eval_data)

            print(f"Epoch {self.best_eval_epoch}, Best evaluation report saved to {save_path}")
            # 保存混淆矩阵图像
            confusion_matrix_path = f"./{self.model_name}_output/best_confusion_matrix.pdf"  # 自定义保存路径
            img.savefig(confusion_matrix_path, format='pdf')
            print(f"Best confusion matrix saved to: {confusion_matrix_path}")
            plt.close(img.figure)  # 关闭图像，释放内存


    def plot_cm(self, cm_metric, gesture_names, fig_size=(10, 7), cmap="Spectral", normalize=True):
        # Assuming GestureClass is an Enum, convert it to a list of names or values
        cm = cm_metric.cpu().numpy()
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # 对每行归一化，避免除以零

        df_cm = pd.DataFrame(
            cm,
            index=gesture_names,  # Assuming this is the row index
            columns=gesture_names  # Convert enum members to list of names
        )
        
        # Plot the confusion matrix using seaborn
        plt.figure(figsize=fig_size)
        fig = sns.heatmap(df_cm, annot=True, cmap=cmap, fmt=".2f").get_figure()

        return fig



