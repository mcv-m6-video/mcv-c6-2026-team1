import argparse
import os
import numpy as np
import logging
from src.detection.runner import build_model
from src.detection.prepare_datasets import get_dataset_dir

def setup_logger(log_file):
    """Configures the logger to write to a specific file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger("CrossValidation")
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Formatting
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    
    logger.addHandler(fh)
    
    return logger

def main(args):
    # Count folds
    data_dir = get_dataset_dir()
    seq_dir = os.path.join(data_dir, "sequential")
    rand_dir = os.path.join(data_dir, "random")
    if not os.path.exists(seq_dir) or not os.path.exists(rand_dir):
        raise FileNotFoundError("Dataset directories do not exist. Generate the folds first.")
        
    seq_folds = sum(1 for d in os.listdir(seq_dir) if d.startswith("fold_") and os.path.isdir(os.path.join(seq_dir, d)))
    rand_folds = sum(1 for d in os.listdir(rand_dir) if d.startswith("fold_") and os.path.isdir(os.path.join(rand_dir, d)))
    
    if seq_folds != rand_folds:
        raise ValueError(f"Fold mismatch! Sequential: {seq_folds} folds | Random: {rand_folds} folds.")
    if seq_folds == 0:
        raise ValueError("No folds found in the dataset directories.")
        
    K = seq_folds

    # Initialize log
    log_file = "src/detection/logs/k_fold_cross_validation.log"
    logger = setup_logger(log_file)
    logger.info("=================================================")
    logger.info(f"Starting YOLO K-Fold Cross-Validation (K={K})")
    logger.info(f"Hyperparameters -> LR: {args.lr} | Batch: {args.batch_size} | Freeze Strategy: {args.freeze_strategy}")
    logger.info("=================================================")

    strategies = {"sequential": "B", "random": "C"}
    
    final_results = {}

    for strategy, letter in strategies.items():
        logger.info(f"\nEvaluating Strategy {letter} ({strategy})")
        fold_scores = []
        
        for k in range(1, K+1):
            fold_dir = os.path.join(data_dir, strategy, f"fold_{k}")
            args.data_dir = fold_dir
            run_name = f"cv_{strategy}_fold{k}"
            
            logger.info(f"  -> Training {run_name}...")
            
            # Initialize fresh model for every fold
            model = build_model("yolo")
            
            # Train the model
            model.train(args)
            
            # Evaluate
            logger.info(f"  -> Validating {run_name}...")
            ap50 = model.evaluate()
            fold_scores.append(ap50)
            
            logger.info(f"  -> {run_name} AP50: {ap50:.4f}")

        # Compute cross-validation
        mean_ap50 = np.mean(fold_scores)
        std_ap50 = np.std(fold_scores)
        final_results[strategy] = {"mean": mean_ap50, "std": std_ap50, "scores": fold_scores}
        
        logger.info("-------------------------------------------------")
        logger.info(f"STRATEGY {letter} RESULTS:")
        logger.info(f"Scores: {[f'{s:.4f}' for s in fold_scores]}")
        logger.info(f"Mean AP50: {mean_ap50:.4f}")
        logger.info(f"Std Dev  : {std_ap50:.4f}")
        logger.info("-------------------------------------------------")

    # Final Summary
    logger.info("\n=================================================")
    logger.info("FINAL CROSS-VALIDATION SUMMARY")
    logger.info("=================================================")
    for strategy in strategies:
        mean = final_results[strategy]["mean"]
        std = final_results[strategy]["std"]
        logger.info(f"Strategy {strategy.capitalize():<12} -> AP50: {mean:.4f} ± {std:.4f}")
    logger.info(f"Logs successfully saved to {log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Best configuration from task 1.2
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--freeze_strategy", type=int, default=2)
    args = parser.parse_args()
    main(args)