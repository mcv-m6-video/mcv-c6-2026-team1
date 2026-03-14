import os
import glob
import argparse
from src.video_utils import load_video
from src.eval import readData, eval, get_sequence_dir, get_gt_data
from src.detection.run_detection import load_best_model, run_detection
from src.optical_flow.runner import build_flow_model
from src.tracking.run_tracking import run_tracking

PRED_FILENAME = "track1.txt"
VIDEO_FILENAME = "tracking.mp4"

def parse_args():
    parser = argparse.ArgumentParser(description="Run single-camera tracking and evaluation on the CVPR 2022 AI City Challenge Track1 dataset.")
    parser.add_argument("seq_id", type=int, choices=[1,3,4], help="Sequence ID (choices: 1, 3, 4)")
    parser.add_argument("-e", "--execute", action="store_true", help="If set, run the tracking algorithm before evaluation.")

    # Tracking and matching
    parser.add_argument("-t", "--tracking_model", type=str, default="overlap", choices=["overlap", "kalman"])
    parser.add_argument("-m", "--matching", type=str, default="greedy", choices=["greedy", "hungarian"])

    # Detection filters
    parser.add_argument("--min_confidence", type=float, default=0.4)

    # Tracking params
    parser.add_argument("--min_iou", type=float, default=0.4)
    parser.add_argument("--max_age", type=int, default=3)

    # Save video w/ tracking
    parser.add_argument("-s", "--save_video", action="store_true", help="Whether to save the output video with the tracking.")

    # Optical flow
    parser.add_argument("-f", "--use_flow", action="store_true", help="Use optical flow in the tracking.")
    parser.add_argument("--flow_method", type=str, default="memflow_kitti", choices=["memflow_kitti", "memflow_sintel", "gmflow", "pyflow"])
    parser.add_argument("--flow_alpha", type=float, default=0.5, help="0.5 mean between Kalman prediction and Optical Flow, 1.0 just Kalman.")
    return parser.parse_args()

def run_tracking_single_cam(args, seq_id, result_dir):
    """
    Placeholder for the actual tracking inference.
    It reads cameras from the sequence path and generates predictions.
    """
    print(f"--- Running Tracking for S{seq_id:02d} ---")
    os.makedirs(result_dir, exist_ok=True)

    seq_dir = get_sequence_dir(seq_id)
    cam_folders = sorted(glob.glob(os.path.join(seq_dir, "c*")))
    if not cam_folders:
        print(f"No camera folders found in {seq_dir}.")
        return
    
    # Detection and flow models
    det_model = load_best_model()
    flow_model, flow_cfg, device = build_flow_model(method=args.flow_method)
    flow_model_dict = {
        "method": args.flow_method,
        "model": flow_model,
        "cfg": flow_cfg,
        "device": device
    }

    # Process all the cameras for the sequence
    for cam_folder in cam_folders:
        cam_str = os.path.basename(cam_folder)
        cid = int(cam_str[1:])
        pred_dir = os.path.join(result_dir, cam_str)
        os.makedirs(pred_dir, exist_ok=True)
        video_file = os.path.join(pred_dir, VIDEO_FILENAME)
        pred_file = os.path.join(pred_dir, PRED_FILENAME)
        print(f"Processing camera {cam_str}...")
        
        input_path = os.path.join(cam_folder, "vdo.avi")
        video = load_video(input_path)

        # Predict bboxes
        preds_by_frame = run_detection(video, det_model)

        # Tracking (w/ optical flow + save to .txt if applicable)
        run_tracking(
            args=args, 
            preds_by_frame=preds_by_frame, 
            video_frames=video, 
            flow_model_dict=flow_model_dict, 
            input_video_path=input_path,
            save_video_path=video_file,
            txt_path=pred_file, 
            cam_id=cid)
        
        print(f"Saved prediction to {pred_file}")

def run_evaluation(seq_id, result_dir):
    """
    Evaluates predictions by directly importing and calling the eval logic.
    """
    print(f"--- Running Tracking Evaluation for S{seq_id:02d} ---")
    pred_dirs = sorted(glob.glob(os.path.join(result_dir, "c*")))
    if not pred_dirs:
        print(f"No prediction files found in {result_dir}. Run with -e first.")
        return

    # Read GT data
    test_df = get_gt_data()

    results = {}
    for pred_dir in pred_dirs:
        cid = int(os.path.basename(pred_dir)[1:])
        cam_str = f"c{cid:03d}"
        print(f"Evaluating {cam_str}...")
        
        try:
            # Read predictions
            pred_df = readData(os.path.join(pred_dir, PRED_FILENAME))
            
            # 2. Compute the metrics directly
            summary = eval(test_df, pred_df, cid=cid)
            
            # 3. Store the native dictionary results
            if summary:
                results[cam_str] = {
                    "IDF1": summary.get("IDF1", 0.0),
                    "HOTA": summary.get("HOTA", 0.0)
                }
            else:
                print(f"Warning: Evaluation returned None for {cam_str}.")
                
        except Exception as e:
            raise Exception(f"Script crashed for {cam_str}: {repr(e)}")

    # Compute Averages and Print Report
    if results:
        print("\n" + "="*40)
        print(f"FINAL EVALUATION REPORT: S{seq_id:02d}")
        print("="*40)
        print(f"{'Camera':<10} | {'IDF1':<10} | {'HOTA':<10}")
        print("-" * 35)
        
        total_idf1 = 0
        total_hota = 0
        
        for cam, metrics in results.items():
            idf1 = metrics['IDF1'] * 100
            hota = metrics['HOTA'] * 100
            total_idf1 += idf1
            total_hota += hota
            print(f"{cam:<10} | {idf1:>6.2f}%   | {hota:>6.2f}%")
            
        print("-" * 35)
        avg_idf1 = total_idf1 / len(results)
        avg_hota = total_hota / len(results)
        print(f"{'Average':<10} | {avg_idf1:>6.2f}%   | {avg_hota:>6.2f}%")
        print("="*40)
        
        # Save summary to file
        summary_file = os.path.join(result_dir, "eval_summary.txt")
        with open(summary_file, "w") as f:
            f.write("Camera,IDF1,HOTA\n")
            for cam, metrics in results.items():
                f.write(f"{cam},{metrics['IDF1']*100:.2f},{metrics['HOTA']*100:.2f}\n")
            f.write(f"Avg.,{avg_idf1:.2f},{avg_hota:.2f}\n")
        print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    args = parse_args()
    
    seq_id = args.seq_id
    result_dir = f"MTSC/S{seq_id:02d}"
    
    if args.execute:
        run_tracking_single_cam(args, seq_id, result_dir)
    
    run_evaluation(seq_id, result_dir)