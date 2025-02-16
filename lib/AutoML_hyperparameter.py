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

# Mapping from dataset names to corresponding gesture lists
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
        Initialize hyperparameter optimizer
        Args:
            train_loader: train data loader
            test_loader: test data loader
            model_name: model name ("quartznet", "quartznet_large")
            dataset_name: dataset name ("UCI_HAR", "SHREC", "DB4", "MHEALTH", "OPPO", "BERKELEY", "LMDHG")
            n_trials: number of optimization attempts
        """
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.n_trials = n_trials
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get the list of gestures corresponding to the dataset
        if dataset_name not in DATASET_TO_GESTURES:
            raise ValueError(f"Dataset type not supported: {dataset_name}")
        self.gesture_list = DATASET_TO_GESTURES[dataset_name]
        
        # Extract dataset features
        self.num_classes = len(self.gesture_list)  # class number
        self.dataset_size = len(train_loader.dataset)  # sample number
        self.input_shape = next(iter(train_loader))[0].shape  # input dimension
        
        # Set up the configuration space
        self.configspace = self._setup_configspace()
        
    def _setup_configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        
        # Dynamically adjust the hyperparameter range based on the dataset size
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

        # QuartzNet specialized parameters
        if self.model_name == "quartznet":
            cs.add([
                UniformIntegerHyperparameter("num_blocks", 3, 7, default_value=5),
                UniformIntegerHyperparameter("num_cells", 3, 7, default_value=5),
                UniformIntegerHyperparameter("input_channels", 128, 512, default_value=256),
                UniformIntegerHyperparameter("head_channels", 256, 1024, default_value=512),
                UniformIntegerHyperparameter("input_kernel", 21, 45, default_value=33),
                UniformIntegerHyperparameter("head_kernel", 63, 99, default_value=87),
            ])
        
    
    def train(self, config: Configuration, seed: int = 42) -> float:
        """Train and evaluate the configuration"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Recreate the data loader using the configured batch size
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
        
        # Obtain the data shape
        sample_data, _ = next(iter(train_loader))
        input_shape = sample_data.shape
        output_shape = len(self.gesture_list)
        
        # Create the model and pass in the correct parameters
        model_params = {
            "dropout_rate": config["dropout_rate"]
        }
        
        if self.model_name == "quartznet":
            model_params.update({
                # Convert the parameter to an integer
                "num_blocks": int(config["num_blocks"]),
                "num_cells": int(config["num_cells"]),
                "input_channels": int(config["input_channels"]),
                "head_channels": int(config["head_channels"]),
                "input_kernel": int(config["input_kernel"]),
                "head_kernel": int(config["head_kernel"])
            })
        
        model = prepare_model(
            model_type=self.model_name,
            input_shape=input_shape,
            output_shape=output_shape,
            device=self.device,
            **model_params
        )
        
        # Set up the training components
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
        
        # Set evaluation metric
        eval_accuracy = MulticlassAccuracy(device=self.device)
        
        # Training loop
        best_accuracy = 0.0
        for epoch in range(5):  # Reduce the number of epochs appropriately to accelerate the optimization process
            model.train()
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(self.device).float()
                batch_labels = batch_labels.to(self.device).long()
                
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
            
            # Evaluate
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
        
        return -best_accuracy  # Return negative values to solve the minimization problem
    
    def optimize(self):
        """Run hyperparameter optimization"""
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
        
        # Save the optimal configuration
        best_config = dict(incumbent)
        
        save_dir = f"./automl_results/{self.dataset_name}/{self.model_name}"
        os.makedirs(save_dir, exist_ok=True)
        
        config_path = os.path.join(save_dir, "best_config.json")
        with open(config_path, "w") as f:
            json.dump(best_config, f, indent=4)
        
        return best_config

def load_best_config(model_name, dataset_name):
    """Load the optimal configuration for the specific model and dataset"""
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
    Retrieve the best hyperparameters. If they already exist, load them directly; otherwise, perform optimization
    Args:
        train_loader: Training data loader
        test_loader: Test data loader
        model_name: Model name (“quartznet”, “large_quartznet”)
        dataset_name: Dataset name
        force_optimize: Whether to force re-optimization even if a saved configuration exists
    Returns:
        dict: Best hyperparameter configuration
    """
    if not force_optimize:
        best_config = load_best_config(model_name, dataset_name)
        if best_config is not None:
            print(f"The best configuration of the {model_name} model has been loaded on the {dataset_name} dataset")
            return best_config
    
    print(f"Starting hyperparameter optimization for the {model_name} model on the {dataset_name} dataset…")
    optimizer = ModelHyperOptimizer(
        train_loader=train_loader,
        test_loader=test_loader,
        model_name=model_name,
        dataset_name=dataset_name,
        n_trials=50
    )
    best_config = optimizer.optimize()
    print(f"Hyperparameter optimization for the {model_name} model on the {dataset_name} dataset is completed")
    return best_config

def main():
    """main function for testing"""
    pass

if __name__ == "__main__":
    main()