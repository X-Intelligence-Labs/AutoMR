from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter
)
import numpy as np
from smac import HyperparameterOptimizationFacade, Scenario
import torch
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy
import json
import os

from dataset_module import (
    UCI_HAR_NAME, SHREC_NAME, 
    DB4_NAME, MHEALTH_NAME, OPPO_NAME,BERKELEY_NAME,
    LMDHG_NAME,
)
from model_factory_module import prepare_model

# 数据集名称到对应手势列表的映射
DATASET_TO_GESTURES = {
    "UCI_HAR": UCI_HAR_NAME,
    "SHREC": SHREC_NAME,
    "DB4": DB4_NAME,
    "MHEALTH": MHEALTH_NAME,
    "OPPO": OPPO_NAME,
    "BERKELEY": BERKELEY_NAME,
    "LMDHG": LMDHG_NAME,

}

class ModelHyperOptimizer:
    def __init__(self, train_loader, test_loader, model_name="quartznet", dataset_name="UCI_HAR", n_trials=50):
        """
        初始化超参数优化器
        Args:
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            model_name: 模型名称 ("quartznet", "cnn")
            dataset_name: 数据集名称 ("UCI_HAR", "SHREC", "DB4", "MHEALTH")
            n_trials: 优化尝试次数
        """
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.n_trials = n_trials
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 获取数据集对应的手势列表
        if dataset_name not in DATASET_TO_GESTURES:
            raise ValueError(f"不支持的数据集名称: {dataset_name}")
        self.gesture_list = DATASET_TO_GESTURES[dataset_name]
        
        # 提取数据集特性
        self.num_classes = len(self.gesture_list)  # 类别数
        self.dataset_size = len(train_loader.dataset)  # 样本数
        self.input_shape = next(iter(train_loader))[0].shape  # 输入维度
        
        # 设置配置空间
        self.configspace = self._setup_configspace()
        
    def _setup_configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        
        # 根据数据集动态调整超参数范围
        if self.dataset_size < 5000:
            lr_upper = 1e-3
            batch_size_upper = 64
        else:
            lr_upper = 1e-2
            batch_size_upper = 256

        cs.add([
            UniformFloatHyperparameter("learning_rate", 1e-4, lr_upper, default_value=1e-3, log=True),
            UniformFloatHyperparameter("weight_decay", 1e-6, 1e-3, default_value=1e-4, log=True),
            UniformFloatHyperparameter("dropout_rate", 0.1, 0.5, default_value=0.25),
            UniformIntegerHyperparameter("batch_size", 16, batch_size_upper, default_value=64, log=True),
        ])

        # QuartzNet特定参数
        if self.model_name == "quartznet":
            cs.add([
                UniformIntegerHyperparameter("num_blocks", 3, 7, default_value=5),
                UniformIntegerHyperparameter("num_cells", 3, 7, default_value=5),
                UniformIntegerHyperparameter("input_channels", 128, 512, default_value=256),
                UniformIntegerHyperparameter("head_channels", 256, 1024, default_value=512),
                UniformIntegerHyperparameter("input_kernel", 21, 45, default_value=33),
                UniformIntegerHyperparameter("head_kernel", 63, 99, default_value=87),
            ])
        elif self.model_name == "cnn":
            # CNN模型特定参数
            cs.add([
                UniformIntegerHyperparameter("num_filters", 8, 64, default_value=32),
                UniformIntegerHyperparameter("num_layers", 2, 6, default_value=4),
            ])
        
        return cs
    
    def train(self, config: Configuration, seed: int = 42) -> float:
        """训练并评估一个配置"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 使用配置的batch_size重新创建数据加载器
        train_dataset = self.train_loader.dataset
        test_dataset = self.test_loader.dataset
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=2
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=2
        )
        
        # 获取数据形状
        sample_data, _ = next(iter(train_loader))
        input_shape = sample_data.shape
        output_shape = len(self.gesture_list)
        
        # 创建模型，传入正确的参数
        model_params = {
            "dropout_rate": config["dropout_rate"]
        }
        
        if self.model_name == "quartznet":
            model_params.update({
                # 将参数转换为整数
                "num_blocks": int(config["num_blocks"]),
                "num_cells": int(config["num_cells"]),
                "input_channels": int(config["input_channels"]),
                "head_channels": int(config["head_channels"]),
                "input_kernel": int(config["input_kernel"]),
                "head_kernel": int(config["head_kernel"])
            })
        elif self.model_name == "cnn":
            model_params.update({
                "num_filters": int(config["num_filters"]),
                "num_layers": int(config["num_layers"])
            })
        
        model = prepare_model(
            model_type=self.model_name,
            input_shape=input_shape,
            output_shape=output_shape,
            device=self.device,
            **model_params
        )
        
        # 设置训练组件
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
        
        # 设置评估指标
        eval_accuracy = MulticlassAccuracy(device=self.device)
        
        # 训练循环
        best_accuracy = 0.0
        for epoch in range(5):  # 适当减少epoch数，加快优化速度
            model.train()
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(self.device).float()
                batch_labels = batch_labels.to(self.device).long()
                
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
            
            # 评估
            model.eval()
            with torch.no_grad():
                for batch_data, batch_labels in test_loader:
                    batch_data = batch_data.to(self.device).float()
                    batch_labels = batch_labels.to(self.device).long()
                    outputs = model(batch_data)
                    eval_accuracy.update(outputs, batch_labels)
                
                accuracy = eval_accuracy.compute().item()
                best_accuracy = max(best_accuracy, accuracy)
                eval_accuracy.reset()
        
        return -best_accuracy  # 返回负值以适应最小化问题
    
    def optimize(self):
        """运行超参数优化"""
        scenario = Scenario(
            configspace=self.configspace,
            n_trials=self.n_trials,
            deterministic=True,
            objectives=["quality"],
            output_directory=f"./smac3_output/{self.dataset_name}/{self.model_name}"
        )
        
        smac = HyperparameterOptimizationFacade(
            scenario=scenario,
            target_function=self.train,
            overwrite=True
        )
        
        incumbent = smac.optimize()
        
        # 保存最佳配置
        best_config = dict(incumbent)
        
        save_dir = f"./automl_results/{self.dataset_name}/{self.model_name}"
        os.makedirs(save_dir, exist_ok=True)
        
        config_path = os.path.join(save_dir, "best_config.json")
        with open(config_path, "w") as f:
            json.dump(best_config, f, indent=4)
        
        return best_config

def load_best_config(model_name, dataset_name):
    """加载特定模型和数据集的最佳配置"""
    config_path = f"./automl_results/{dataset_name}/{model_name}/best_config.json"
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def get_best_hyperparameters(
    train_loader, 
    test_loader, 
    model_name, 
    dataset_name,
    force_optimize=False
):
    """
    获取最佳超参数，如果已存在则直接加载，否则运行优化
    Args:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        model_name: 模型名称 ("quartznet", "cnn")
        dataset_name: 数据集名称
        force_optimize: 是否强制重新优化，即使存在已保存的配置
    Returns:
        dict: 最佳超参数配置
    """
    if not force_optimize:
        best_config = load_best_config(model_name, dataset_name)
        if best_config is not None:
            print(f"已加载{dataset_name}数据集上{model_name}模型的最佳配置")
            return best_config
    
    print(f"开始为{dataset_name}数据集上的{model_name}模型优化超参数...")
    optimizer = ModelHyperOptimizer(
        train_loader=train_loader,
        test_loader=test_loader,
        model_name=model_name,
        dataset_name=dataset_name,
        n_trials=50
    )
    best_config = optimizer.optimize()
    print(f"{dataset_name}数据集上的{model_name}模型超参数优化完成")
    return best_config

def main():
    """主函数，用于测试"""
    pass

if __name__ == "__main__":
    main()