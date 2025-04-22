# main.py
import os
import sys
import argparse
from datetime import datetime
import torch

from utils.logger import setup_logging, get_logger
from utils.config import load_config
from utils.utils import set_seed
from models import get_model
from datasets import get_dataset
from core import Trainer, Evaluator, Searcher
from deploy.inference import Predictor

# Initialize logger
logger = get_logger(__name__)

def train(config, args):
    """Run training process."""
    logger.info("Starting training process")
    
    # Set random seed for reproducibility
    if "seed" in config:
        set_seed(config["seed"])
        
    # Setup experiment directory and logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config.get("experiment_name", "experiment")
    exp_dir = os.path.join(config.get("output_dir", "outputs"), f"{exp_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Set up logging
    setup_logging(os.path.join(exp_dir, "logs"))
    
    # Update config with experiment directory
    config["save_dir"] = os.path.join(exp_dir, "checkpoints")
    
    # Get dataset
    dataset_name = config.get("dataset_name", "cifar10")
    dataset_config = config.get("dataset_config", {})
    dataset = get_dataset(dataset_name, dataset_config)
    
    # Prepare and setup data
    dataset.prepare_data()
    dataset.setup()
    
    # Create data loaders
    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Setup trainer
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.setup(train_loader=train_loader, val_loader=val_loader)
        start_epoch = trainer.load_checkpoint(args.resume)
    else:
        # Get model config
        num_classes = dataset.get_num_classes()
        input_shape = dataset.get_input_shape()
        
        model_config = config.get("model_config", {})
        model_config["num_classes"] = num_classes
        model_config["img_size"] = input_shape[1]  # Height
        
        logger.info(f"Model config: {model_config}")
        
        # Initialize model directly
        trainer.setup(train_loader=train_loader, val_loader=val_loader)
        start_epoch = 0
        
    # Run training
    trainer.run(start_epoch=start_epoch)
    
    # Evaluate on test set if available
    if dataset.test_dataloader() is not None:
        logger.info("Evaluating on test set")
        evaluator = Evaluator(config)
        evaluator.setup(model=trainer.model, test_loader=dataset.test_dataloader())
        metrics = evaluator.run()
        
        # Log metrics
        for metric, value in metrics.items():
            if metric != "confusion_matrix":
                logger.info(f"Test {metric}: {value:.4f}")
                
    logger.info(f"Training completed. Results saved to {exp_dir}")
    return exp_dir

def search(config, args):
    """Run hyperparameter search."""
    logger.info("Starting hyperparameter search")
    
    # Set random seed for reproducibility
    if "seed" in config:
        set_seed(config["seed"])
        
    # Setup experiment directory and logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config.get("experiment_name", "hparam_search")
    exp_dir = os.path.join(config.get("output_dir", "outputs"), f"{exp_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Set up logging
    setup_logging(os.path.join(exp_dir, "logs"))
    
    # Update config with experiment directory
    config["save_dir"] = os.path.join(exp_dir, "results")
    
    # Get dataset
    dataset_name = config.get("dataset_name", "cifar10")
    dataset_config = config.get("dataset_config", {})
    dataset = get_dataset(dataset_name, dataset_config)
    
    # Prepare and setup data
    dataset.prepare_data()
    dataset.setup()
    
    # Create data loaders
    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()
    test_loader = dataset.test_dataloader()
    
    # Update config with dataset info
    config["num_classes"] = dataset.get_num_classes()
    config["input_shape"] = dataset.get_input_shape()
    
    # Initialize searcher
    searcher = Searcher(config)
    
    # Setup searcher
    searcher.setup(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
    
    # Run search
    results = searcher.run()
    
    logger.info(f"Hyperparameter search completed. Results saved to {config['save_dir']}")
    return results

def evaluate(config, args):
    """Run evaluation on a trained model."""
    logger.info("Starting evaluation process")
    
    # Check if model path is provided
    if not args.model_path:
        logger.error("Model path must be provided for evaluation")
        sys.exit(1)
        
    # Set random seed for reproducibility
    if "seed" in config:
        set_seed(config["seed"])
        
    # Set up logging
    setup_logging(os.path.join(config.get("output_dir", "outputs"), "logs"))
    
    # Get dataset
    dataset_name = config.get("dataset_name", "cifar10")
    dataset_config = config.get("dataset_config", {})
    dataset = get_dataset(dataset_name, dataset_config)
    
    # Prepare and setup data
    dataset.prepare_data()
    dataset.setup()
    
    # Create test loader
    test_loader = dataset.test_dataloader()
    
    if test_loader is None:
        logger.error("Test loader is not available")
        sys.exit(1)
        
    # Update config with checkpoint path
    config["checkpoint_path"] = args.model_path
    
    # Initialize evaluator
    evaluator = Evaluator(config)
    
    # Setup evaluator (will load model from checkpoint)
    evaluator.setup(test_loader=test_loader)
    
    # Run evaluation
    metrics = evaluator.run()
    
    # Log metrics
    for metric, value in metrics.items():
        if metric != "confusion_matrix":
            logger.info(f"{metric.capitalize()}: {value:.4f}")
            
    return metrics

def infer(config, args):
    """Run inference with a trained model."""
    logger.info("Starting inference process")
    
    # Check if model path and input are provided
    if not args.model_path:
        logger.error("Model path must be provided for inference")
        sys.exit(1)
        
    if not args.input:
        logger.error("Input image path must be provided for inference")
        sys.exit(1)
        
    # Set up logging
    setup_logging(os.path.join(config.get("output_dir", "outputs"), "logs"))
    
    # Initialize predictor
    predictor = Predictor(args.model_path, config=config)
    
    # Run inference
    if os.path.isdir(args.input):
        # Batch inference on directory
        image_files = [os.path.join(args.input, f) for f in os.listdir(args.input) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        logger.info(f"Running batch inference on {len(image_files)} images")
        
        if args.return_probs:
            predictions = predictor.batch_predict(image_files, return_probs=True)
            
            # Print top predictions for each image
            for i, probs in enumerate(predictions):
                top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:5]
                logger.info(f"Image {i+1} ({os.path.basename(image_files[i])}):")
                for idx in top_indices:
                    logger.info(f"  Class {idx}: {probs[idx]:.4f}")
        else:
            predictions = predictor.batch_predict(image_files)
            
            # Print predictions
            for i, pred in enumerate(predictions):
                logger.info(f"Image {i+1} ({os.path.basename(image_files[i])}): Class {pred}")
    else:
        # Single image inference
        logger.info(f"Running inference on {args.input}")
        
        if args.return_probs:
            probs = predictor.predict(args.input, return_probs=True)
            
            # Print top predictions
            top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:5]
            for idx in top_indices:
                logger.info(f"Class {idx}: {probs[idx]:.4f}")
        else:
            prediction = predictor.predict(args.input)
            logger.info(f"Prediction: Class {prediction}")
            
    return predictions

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Vision Transformer training, evaluation, and inference")
    
    # Mode arguments
    parser.add_argument("mode", type=str, choices=["train", "search", "eval", "infer"],
                       help="Operation mode: train, hyperparameter search, evaluate, or infer")
    
    # Configuration
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    
    # Training arguments
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume training from")
    
    # Evaluation/Inference arguments
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to model checkpoint for evaluation or inference")
    parser.add_argument("--input", type=str, default=None,
                       help="Path to input image or directory for inference")
    parser.add_argument("--return-probs", action="store_true",
                       help="Return class probabilities instead of class prediction")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.error(f"Config file {args.config} not found")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        sys.exit(1)
    
    # Run the selected mode
    if args.mode == "train":
        train(config, args)
    elif args.mode == "search":
        search(config, args)
    elif args.mode == "eval":
        evaluate(config, args)
    elif args.mode == "infer":
        infer(config, args)
    else:
        logger.error(f"Invalid mode: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()