import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib import cm

from src.re_id.projector import SpatioTemporalProjector
from src.eval import AI_CITY_DATA_DIR

def visualize_gt_projection_final(seq_id, cam_id):
    seq_str = f"S{seq_id:02d}"
    cam_str = f"c{cam_id:03d}"
    
    gt_path = os.path.join(AI_CITY_DATA_DIR, "train", seq_str, cam_str, "gt", "gt.txt")
    if not os.path.exists(gt_path):
        return

    # 1. Parse GT
    frame_detections = defaultdict(list)
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6: continue
            fid, oid, x, y, w, h = map(float, parts[:6])
            frame_detections[int(fid)].append({'id': int(oid), 'bbox': [x, y, x + w, y + h]})

    # 2. Find frame with >= 4 vehicles
    target_fid = next(fid for fid, dets in sorted(frame_detections.items()) if len(dets) >= 4)
    target_dets = frame_detections[target_fid]

    # 3. Load Video Frame
    video_path = os.path.join(AI_CITY_DATA_DIR, "train", seq_str, cam_str, "vdo.avi")
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_fid - 1)
    ret, frame = cap.read()
    cap.release()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 4. Initialize Projector
    projector = SpatioTemporalProjector(seq_id)
    
    # Generate distinct colors for each ID
    unique_ids = [d['id'] for d in target_dets]
    color_map = cm.get_cmap('tab10', len(unique_ids))
    id_to_color = {oid: tuple(np.array(color_map(i)[:3]) * 255) for i, oid in enumerate(unique_ids)}

    projected_points = []
    
    # 5. Draw and Project
    for det in target_dets:
        oid = det['id']
        color = id_to_color[oid]
        x1, y1, x2, y2 = map(int, det['bbox'])
        
        # INCREASED: BBox thickness set to 6
        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 6)
        
        # INCREASED: Label font scale set to 1.5 and thickness to 4
        cv2.putText(frame_rgb, f"ID {oid}", (x1, y1-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
        
        # INCREASED: Projection anchor sizes
        xc, yb = (x1 + x2) // 2, y2
        cv2.circle(frame_rgb, (xc, yb), 14, (255, 255, 255), -1)
        cv2.circle(frame_rgb, (xc, yb), 10, color, -1)

        # Get GPS (Long, Lat)
        coord = projector.get_ground_plane_coord(cam_id, det['bbox'])
        projected_points.append({'id': oid, 'coord': coord, 'color': np.array(color)/255.0})

    # 6. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(24, 11))
    
    axes[0].imshow(frame_rgb)
    axes[0].set_title(f"Camera View: {seq_str} {cam_str} | Frame {target_fid}", fontsize=20, fontweight='bold')
    axes[0].axis('off')

    # GPS Plane
    axes[1].set_title("Projected GPS Coordinates", fontsize=20, fontweight='bold')
    axes[1].set_xlabel("Longitude", fontsize=16)
    axes[1].set_ylabel("Latitude", fontsize=16)
    
    for pt in projected_points:
        # INCREASED: Scatter marker size (s) set to 600
        axes[1].scatter(pt['coord'][1], pt['coord'][0], color=pt['color'], s=600, 
                        edgecolors='black', linewidth=2, label=f"ID {pt['id']}")
        
        # INCREASED: Annotation font size and offset
        axes[1].annotate(f"ID {pt['id']}", (pt['coord'][1], pt['coord'][0]), 
                         xytext=(15, 15), textcoords='offset points', 
                         fontsize=16, fontweight='bold')

    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    plt.savefig(f"presentation_projection_S{seq_id}_C{cam_id}.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    # Example for Sequence 1, Camera 4
    visualize_gt_projection_final(seq_id=1, cam_id=4)