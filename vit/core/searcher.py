import os
import json
import torch
from typing import Any, Optional, Tuple
import numpy as np
from tqdm import tqdm
import optuna
from optuna.samplers import TPESampler
from optuna.trial import Trial

from .base import BaseHandler
from .trainer import Trainer
from .evaluator import Evaluator
from ..utils.logger import get_logger

logger = get_logger(__name__)

class Searcher(BaseHandler):
    """Hyperparameter search using Optuna."""
    
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.n_trials = config.get("n_trials", 10)
        self.study_name = config.get("study_name", "vit_hparam_search")
        self.storage = config.get("storage", None)
        self.search_space = config.get("search_space", {})
        self.save_dir = config.get("save_dir", "hparam_search")
        self.metric = config.get("metric", "val_loss")
        self.direction = config.get("direction", "minimize")
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
    def setup(self, train_loader, val_loader, test_loader=None):
        """Setup data loaders for search."""
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
    def define_model_params(self, trial: Trial) -> dict[str, Any]:
        """Define model hyperparameters to search."""
        model_params = {}
        
        if "model" in self.search_space:
            model_space = self.search_space["model"]
            
            # Vision Transformer specific parameters
            if "img_size" in model_space:
                options = model_space["img_size"]["options"]
                model_params["img_size"] = trial.suggest_categorical("img_size", options)
                
            if "patch_size" in model_space:
                options = model_space["patch_size"]["options"]
                model_params["patch_size"] = trial.suggest_categorical("patch_size", options)
                
            if "embed_dim" in model_space:
                range_min = model_space["embed_dim"]["min"]
                range_max = model_space["embed_dim"]["max"]
                model_params["embed_dim"] = trial.suggest_int("embed_dim", range_min, range_max, step=16)
                
            if "num_heads" in model_space:
                range_min = model_space["num_heads"]["min"]
                range_max = model_space["num_heads"]["max"]
                model_params["num_heads"] = trial.suggest_int("num_heads", range_min, range_max)
                
            if "depth" in model_space:
                range_min = model_space["depth"]["min"]
                range_max = model_space["depth"]["max"]
                model_params["depth"] = trial.suggest_int("depth", range_min, range_max)
                
            if "mlp_dim" in model_space:
                range_min = model_space["mlp_dim"]["min"]
                range_max = model_space["mlp_dim"]["max"]
                model_params["mlp_dim"] = trial.suggest_int("mlp_dim", range_min, range_max, step=128)
                
        return model_params
    
    def define_training_params(self, trial: Trial) -> dict[str, Any]:
        """Define training hyperparameters to search."""
        training_params = {}
        
        if "training" in self.search_space:
            training_space = self.search_space["training"]
            
            # Optimizer related
            if "learning_rate" in training_space:
                log_min = training_space["learning_rate"]["log_min"]
                log_max = training_space["learning_rate"]["log_max"]
                training_params["learning_rate"] = trial.suggest_float("learning_rate", log_min, log_max, log=True)
                
            if "optimizer" in training_space:
                options = training_space["optimizer"]["options"]
                training_params["optimizer"] = trial.suggest_categorical("optimizer", options)
                
            if "weight_decay" in training_space:
                log_min = training_space["weight_decay"]["log_min"]
                log_max = training_space["weight_decay"]["log_max"]
                training_params["weight_decay"] = trial.suggest_float("weight_decay", log_min, log_max, log=True)
                
            # Scheduler related
            if "scheduler" in training_space:
                options = training_space["scheduler"]["options"]
                training_params["scheduler"] = trial.suggest_categorical("scheduler", options)
                
            # Data related
            if "batch_size" in training_space:
                options = training_space["batch_size"]["options"]
                training_params["batch_size"] = trial.suggest_categorical("batch_size", options)
                
        return training_params
    
    def objective(self, trial: Trial) -> float:
        """Objective function for optimization."""
        # Get trial parameters
        model_params = self.define_model_params(trial)
        training_params = self.define_training_params(trial)
        
        # Create config for this trial
        trial_config = self.config.copy()
        trial_config["model_config"] = model_params
        trial_config.update(training_params)
        
        # Set up trainer
        trainer = Trainer(trial_config)
        trainer.setup(train_loader=self.train_loader, val_loader=self.val_loader)
        
        # Train for specified epochs (reduced for HP search)
        epochs = self.config.get("search_epochs", 10)
        trial_config["epochs"] = epochs
        
        # Train and get validation performance
        trainer.run()
        
        # Return the metric to optimize
        if self.metric == "val_loss":
            return trainer.history["val_loss"][-1]
        elif self.metric == "val_acc":
            return -trainer.history["val_acc"][-1]  # Negative since we want to maximize accuracy
        else:
            raise ValueError(f"Metric {self.metric} not supported for optimization")
    
    def run(self):
        """Run hyperparameter search."""
        if self.train_loader is None or self.val_loader is None:
            raise ValueError("Data loaders must be set up before hyperparameter search")
            
        # Create Optuna study
        direction = "minimize" if self.direction == "minimize" else "maximize"
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=TPESampler(seed=42),
            direction=direction,
            load_if_exists=True
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # Get best parameters and results
        best_params = study.best_params
        best_value = study.best_value
        
        # Log results
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best {self.metric}: {best_value}")
        logger.info("Best hyperparameters:")
        for param, value in best_params.items():
            logger.info(f"    {param}: {value}")
            
        # Save results
        results = {
            "best_params": best_params,
            "best_value": best_value,
            "best_trial": study.best_trial.number
        }
        
        with open(os.path.join(self.save_dir, "best_params.json"), "w") as f:
            json.dump(results, f, indent=4)
            
        # Optionally evaluate best model on test set
        if self.test_loader is not None and self.config.get("evaluate_best", True):
            logger.info("Evaluating best model on test set...")
            
            # Create config with best parameters
            best_config = self.config.copy()
            best_config["model_config"] = {k: v for k, v in best_params.items() 
                                         if k in ["img_size", "patch_size", "embed_dim", 
                                                 "num_heads", "depth", "mlp_dim"]}
            
            # Add training params
            for k, v in best_params.items():
                if k in ["learning_rate", "optimizer", "weight_decay", "scheduler"]:
                    best_config[k] = v
            
            # Train final model with best parameters
            logger.info("Training final model with best parameters...")
            trainer = Trainer(best_config)
            trainer.setup(train_loader=self.train_loader, val_loader=self.val_loader)
            trainer.run()
            
            # Save best model checkpoint
            best_model_path = os.path.join(self.save_dir, "best_model.pth")
            torch.save({
                "model_state_dict": trainer.model.state_dict(),
                "config": best_config
            }, best_model_path)
            
            # Evaluate on test set
            evaluator = Evaluator(best_config)
            evaluator.setup(model=trainer.model, test_loader=self.test_loader)
            test_results = evaluator.run()
            
            # Save test results
            with open(os.path.join(self.save_dir, "test_results.json"), "w") as f:
                json.dump({k: float(v) if isinstance(v, np.ndarray) and v.size == 1 else v.tolist() 
                          if isinstance(v, np.ndarray) else v 
                          for k, v in test_results.items()}, f, indent=4)
            
        return results 