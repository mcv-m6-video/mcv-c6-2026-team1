import os
import glob
import argparse
import cv2
from collections import defaultdict
from src.video_utils import init_video
from src.eval import readData, eval, get_sequence_dir, get_gt_data

from src.re_id.tracker import CityScaleTracker
from src.re_id.trans_re_id import TransReID
from src.re_id.projector import SpatioTemporalProjector
from src.re_id.box_grained import BoxGrainedFilter

MTSC_DIR = "MTSC"
MTMC_DIR = "MTMC"
PRED_FILENAME = "track1.txt"

def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-camera ReID and evaluation for the CVPR 2022 AI City Challenge Track1 dataset.")
    parser.add_argument("seq_id", type=int, choices=[1,3,4], help="Sequence ID (choices: 1, 3, 4)")
    parser.add_argument("-e", "--execute", action="store_true", help="If set, run multi-camera ReID before evaluation.")
    return parser.parse_args()

def parse_mtsc_predictions(txt_path, cam_id):
    """
    Reads the official submission format:
    <camera_id> <obj_id> <frame_id> <xmin> <ymin> <width> <height> <xworld> <yworld>
    Returns a dictionary grouped by obj_id. Ensures camera_id is the same as cam_id
    """
    tracklets = defaultdict(list)
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7: continue
            
            read_cam_id = int(parts[0])
            assert cam_id == read_cam_id, f"Incorrect cam_id in {txt_path} ({read_cam_id} != {cam_id})!"

            obj_id = int(parts[1])
            frame_id = int(parts[2])
            x, y, w, h = map(float, parts[3:7])
            
            bbox_xyxy = [x, y, x + w, y + h]
            tracklets[obj_id].append({
                'frame': frame_id,
                'bbox': bbox_xyxy,
                'area': w * h,
            })
    return tracklets


def build_camera_tracklets(seq_id, cam_id, tracklets_dict, reid_extractor, box_filter, projector):
    """
    Finds the best frame for each tracklet and extracts the ReID features.
    Stores full GPS trajectory and best-frame embedding.
    """
    # 1. Select the best frame representing each tracklet
    frames_to_extract = defaultdict(list)
    final_tracklets = {}
    
    for obj_id, detections in tracklets_dict.items():
        # Sort by bounding box area to find the largest, clearest view
        detections.sort(key=lambda d: d['area'], reverse=True)
        
        # TODO: TRY TOP-K MEAN EMBEDDINGS INSTEAD OF JUST THE BEST FRAME
        best_det = None
        for det in detections:
            if box_filter.is_trustworthy(det['bbox'], cam_id):
                best_det = det
                break
        if best_det is None:
            best_det = detections[0]

        frames_to_extract[best_det['frame']].append({
            'obj_id': obj_id,
            'bbox': best_det['bbox']
        })

        # Build tracklet trajectory
        trajectory = []
        for det in detections:
            gps = projector.get_ground_plane_coord(cam_id, det['bbox'])
            time = projector.get_global_time(cam_id, det['frame'])
            trajectory.append({
                'gps': gps,
                'time': time,
            })
        trajectory = sorted(trajectory, key=lambda x: x['time'])

        final_tracklets[obj_id] = {
            'n_cams': 1,
            'features': None,
            'trajectory': trajectory
        }

    # 2. Extract ReID features in a single video pass
    video_path = os.path.join(get_sequence_dir(seq_id), f"c{cam_id:03d}/vdo.avi")
    cap = init_video(video_path)
    current_frame = 1
    while cap.isOpened():
        ret, frame_img = cap.read()
        if not ret:
            break

        if current_frame in frames_to_extract:
            for req in frames_to_extract[current_frame]:
                x1, y1, x2, y2 = map(int, req['bbox'])
                
                # Crop bounding box in image
                h_img, w_img = frame_img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                crop = frame_img[y1:y2, x1:x2]
                if crop.size == 0:
                    raise ValueError(f"Empty crop for obj_id {req['obj_id']} at frame {current_frame} in Camera {cam_id}!")

                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                features = reid_extractor.extract_features(
                    [crop_rgb],
                    cam_ids=[cam_id],
                )[0]
                final_tracklets[req["obj_id"]]["features"] = features

        current_frame += 1

    cap.release()
    return final_tracklets


def run_mtmc_reid(seq_id, result_dir):
    """
    Executes the global MTMC ReID across all cameras in a sequence.
    """
    print(f"\n--- Starting MTMC ReID for Sequence S{seq_id:02d} ---")
    seq_dir = os.path.join(MTSC_DIR, f"S{seq_id:02d}")
    cam_folders = sorted(glob.glob(os.path.join(seq_dir, "c*")))
    cam_ids_in_sequence = [int(os.path.basename(folder)[1:]) for folder in cam_folders]
    camera_num = max(cam_ids_in_sequence) + 1 # embedding table must be large enough to index by the actual camera id.
    view_num = 1
    if not cam_folders:
        print("No camera predictions found. Run 'single_camera.py' first.")
        return

    # Initialize modules
    projector = SpatioTemporalProjector(seq_id=seq_id)
    box_filter = BoxGrainedFilter(projector.img_sizes)
    reid_extractor = TransReID(
        config_file="./configs/trans_reid_config.yaml",
        model_weights_path="./models/transreid/best_model.pth",
        camera_num=camera_num,
        view_num=view_num,
    )
    tracker = CityScaleTracker()

    # global_id_map: { cam_id: { local_obj_id: global_obj_id } }
    global_id_map = defaultdict(dict)
    global_tracklets = {}
    next_global_id = 1

    # Sequentially match cameras to the global track registry
    for cam_folder in cam_folders:
        cam_str = os.path.basename(cam_folder)
        cam_id = int(cam_str[1:])
        txt_path = os.path.join(cam_folder, PRED_FILENAME)
        
        print(f"Processing local tracks for Camera {cam_str}...")
        local_dict = parse_mtsc_predictions(txt_path, cam_id)
        local_tracklets = build_camera_tracklets(seq_id, cam_id, local_dict, reid_extractor, box_filter, projector)

        if not global_tracklets:
            # First camera initializes the global registry
            for local_id, t in local_tracklets.items():
                global_id_map[cam_id][local_id] = next_global_id
                global_tracklets[next_global_id] = t
                next_global_id += 1

        else:
            # Match current camera against the established global registry
            matches = tracker.associate_tracks(global_tracklets, local_tracklets)
            matched_local_ids = set()
            for global_id, local_id in matches:
                global_id_map[cam_id][local_id] = global_id
                matched_local_ids.add(local_id)
                print(global_tracklets[global_id])
                tracker.merge_tracks(global_tracklets[global_id], local_tracklets[local_id])
                print(global_tracklets[global_id])
                exit(0)

            # Assign new global IDs to unmatched vehicles in this camera
            for local_id, t in local_tracklets.items():
                if local_id not in matched_local_ids:
                    global_id_map[cam_id][local_id] = next_global_id
                    global_tracklets[next_global_id] = t
                    next_global_id += 1

    # Rewrite tracks into a unified MTMC file
    os.makedirs(result_dir, exist_ok=True)
    mtmc_output_file = os.path.join(result_dir, PRED_FILENAME)
    print(f"\nWriting globally consistent IDs to {mtmc_output_file}...")
    with open(mtmc_output_file, 'w') as f_out:
        for cam_folder in cam_folders:
            cam_str = os.path.basename(cam_folder)
            cam_id = int(cam_str[1:])
            txt_path = os.path.join(cam_folder, PRED_FILENAME)
            
            with open(txt_path, 'r') as f_in:
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) < 7: continue
                    
                    # Translate local ID to global ID
                    local_id = int(parts[1])
                    global_id = global_id_map[cam_id][local_id]
                    
                    parts[1] = str(global_id)
                    f_out.write(" ".join(parts) + "\n")
                    
    print("MTMC Association Complete.")


def run_evaluation(seq_id, result_dir):
    """
    Evaluates predictions by directly importing and calling the eval logic.
    """
    print(f"--- Running Tracking Evaluation for S{seq_id:02d} ---")
    if not os.path.exists(result_dir):
        print(f"Directory {result_dir} does not exist. Run with -e first.")
        return

    # Read GT data
    test_df = get_gt_data()

    # Read predictions
    pred_df = readData(os.path.join(result_dir, PRED_FILENAME))
    
    # Compute the metrics directly
    result = eval(test_df, pred_df)
    if result:
        idf1 = result.get("IDF1", 0.0) * 100
        hota = result.get("HOTA", 0.0) * 100

        print("\n" + "="*40)
        print(f"EVALUATION: S{seq_id:02d}")
        print("="*40)
        print(f"{'IDF1':<10} | {'HOTA':<10}")
        print("-" * 35)
        print(f"{idf1:>6.2f}%   | {hota:>6.2f}%")
        print("-" * 35)
        print("="*40)
        
        # Save to file
        file = os.path.join(result_dir, "eval.txt")
        with open(file, "w") as f:
            f.write(f"{'IDF1':<10} | {'HOTA':<10}\n")
            f.write("-" * 35)
            f.write(f"\n{idf1:>6.2f}%   | {hota:>6.2f}%\n")
        print(f"Evaluation saved to {file}")

    else:
        print(f"Warning: Evaluation returned None.")


if __name__ == "__main__":
    args = parse_args()

    seq_id = args.seq_id
    result_dir = os.path.join(MTMC_DIR, f"S{seq_id:02d}")

    if args.execute:
        run_mtmc_reid(seq_id, result_dir)

    run_evaluation(seq_id, result_dir)