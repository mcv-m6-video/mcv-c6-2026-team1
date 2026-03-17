import os
import argparse
import wandb
import numpy as np

# Import your pipeline functions
from src.multi_camera import run_mtmc_reid, MTMC_DIR, PRED_FILENAME
from src.eval import get_gt_data, readData, eval

def parse_args():
    parser = argparse.ArgumentParser(description="Generic W&B Sweep Script for MTMC ReID")
    
    # Catch parameters injected by the W&B agent
    parser.add_argument("--visual_only", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--visual_thr", type=float, default=0.2)
    parser.add_argument("--spatial_thr", type=float, default=35.0)
    
    return parser.parse_args()

def main():
    args = parse_args()

    # wandb.init automatically syncs the args to wandb.config
    wandb.init(config=args)
    config = wandb.config

    sequences = [1, 4]
    idf1_scores = []
    hota_scores = []
    
    # Create a dictionary to hold all metrics for a single log execution
    sweep_metrics = {}

    print(f"\n>>> Sweep Run | Visual Only: {config.visual_only} | Vis Thr: {config.visual_thr} | Spa Thr: {config.spatial_thr} <<<")

    # Load the Ground Truth once for evaluation
    test_df = get_gt_data()

    for seq_id in sequences:
        result_dir = os.path.join(MTMC_DIR, f"S{seq_id:02d}")
        
        # 1. Map the swept parameters to the tracker
        tracker_config = {
            "visual_only": config.visual_only,
            "visual_threshold": config.visual_thr,
            "spatial_threshold": config.spatial_thr 
        }
        
        # 2. Run the tracking pipeline
        run_mtmc_reid(seq_id, result_dir, **tracker_config)

        # 3. Evaluate the results directly
        pred_df = readData(os.path.join(result_dir, PRED_FILENAME))
        result = eval(test_df, pred_df)
        
        if result:
            idf1 = result.get("IDF1", 0.0) * 100
            hota = result.get("HOTA", 0.0) * 100
            idf1_scores.append(idf1)
            hota_scores.append(hota)

            # Add to the dictionary instead of logging immediately
            sweep_metrics[f"S{seq_id:02d}_IDF1"] = idf1
            sweep_metrics[f"S{seq_id:02d}_HOTA"] = hota
            print(f"S{seq_id:02d} -> IDF1: {idf1:.2f}, HOTA: {hota:.2f}")
        else:
            idf1_scores.append(0.0)
            hota_scores.append(0.0)

    # 4. Compute the target mean metrics
    mean_idf1 = float(np.mean(idf1_scores)) if idf1_scores else 0.0
    mean_hota = float(np.mean(hota_scores)) if hota_scores else 0.0
    
    # Add means to the dictionary
    sweep_metrics["mean_IDF1"] = mean_idf1
    sweep_metrics["mean_HOTA"] = mean_hota
    
    # Log everything at once for a perfect single-step WandB record
    wandb.log(sweep_metrics)
    
    print(f"\nSweep Run Complete | Mean IDF1: {mean_idf1:.2f}% | Mean HOTA: {mean_hota:.2f}%")

if __name__ == "__main__":
    main()