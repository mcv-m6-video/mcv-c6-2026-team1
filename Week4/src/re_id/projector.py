import numpy as np
import os
import cv2
from src.eval import AI_CITY_DATA_DIR

H_PREFIX = "Homography matrix:"
K_PREFIX = "Intrinsic parameter matrix:"
D_PREFIX = "Distortion coefficients:"

class SpatioTemporalProjector:
    """
    Uses calibration and timestamps to synchronize cameras from a specific sequence.
    """
    def __init__(self, seq_id):
        def _read_sequence_data(txt_filepath, int_values=False):
            seq_data = {}

            with open(txt_filepath, 'r') as f:
                for line in f:
                    cam_str, value = line.strip().split()
                    cam_id = int(cam_str.replace('c', ''))
                    seq_data[cam_id] = int(value) if int_values else float(value)

            return seq_data

        def _parse_calibration(calib_string):
            calib_data = {'H': None, 'K': None, 'D': None}
            
            for line in calib_string.strip().split('\n'):
                if line.startswith(H_PREFIX):
                    calib_data['H'] = np.array([[float(x) for x in row.split()] for row in line[len(H_PREFIX):].split(";")])
                elif line.startswith(K_PREFIX):
                    calib_data['K'] = np.array([[float(x) for x in row.split()] for row in line[len(K_PREFIX):].split(";")])
                elif line.startswith(D_PREFIX):
                    calib_data['D'] = np.array([float(x) for x in line[len(D_PREFIX):].split()])

            assert (calib_data["K"] is None) == (calib_data["D"] is None), "Mismatch between intrinsics and distortion coefficients!"
            return calib_data

        seq_name = f"S{seq_id:02d}"

        # Parse sequence data
        framenum = _read_sequence_data(os.path.join(AI_CITY_DATA_DIR, "cam_framenum", f"{seq_name}.txt"), True)
        timestamp = _read_sequence_data(os.path.join(AI_CITY_DATA_DIR, "cam_timestamp", f"{seq_name}.txt"))
        assert len(framenum) == len(timestamp), f"Mismatch between number of cameras {len(framenum)} != {len(timestamp)})!"

        # Get camera resolution
        img_sizes = {}
        for cam_id in framenum.keys():
            roi = cv2.imread(os.path.join(AI_CITY_DATA_DIR, "train", seq_name, f"c{cam_id:03d}", "roi.jpg"))
            h, w = roi.shape[:2]
            img_sizes[cam_id] = {"height": h, "width": w}

        # Parse camera calibration
        calibrations = {}
        for cam_id in framenum.keys():
            with open(os.path.join(AI_CITY_DATA_DIR, "train", seq_name, f"c{cam_id:03d}", "calibration.txt"), 'r') as f:
                calibrations[cam_id] = _parse_calibration(f.read())

        # Account for S03_c015 specific FPS
        fps = {}
        for cam_id in framenum.keys():
            fps[cam_id] = 8.0 if seq_id == 3 and cam_id == 15 else 10.0

        self.calibrations = calibrations
        self.img_sizes = img_sizes
        self.fps = fps
        self.timestamp = timestamp

    def get_ground_plane_coord(self, cam_id, bbox):
        """Projects the bottom-center of the bounding box to the GPS ground plane."""
        x_min, y_min, x_max, y_max = bbox
        x_center = (x_min + x_max) / 2.0
        y_bottom = y_max
        
        calib = self.calibrations[cam_id]
        H, K, D = calib['H'], calib['K'], calib['D']
        
        # Removes radial distortion if needed
        if K is not None:
            pts = np.array([[[x_center, y_bottom]]], dtype=np.float32)
            x_center, y_bottom = cv2.undistortPoints(pts, cameraMatrix=K, distCoeffs=D, P=K)[0][0]
            
        # Apply homography to ground plane
        projected = H @ np.array([x_center, y_bottom, 1.0])
        ground_coord = (projected / projected[2])[:2]

        return ground_coord

    def get_global_time(self, cam_id, frame_idx):
        """Converts a local frame index (1-based!) to a global synchronized timestamp across the sequence."""
        return self.timestamp[cam_id] + ((frame_idx-1) / self.fps[cam_id])

if __name__ == "__main__":

    # Unit test: remove distortion of S01_c005
    seq_id = 1
    cam_id = 5
    seq_str = f"S{seq_id:02d}"
    cam_str = f"c{cam_id:03d}"

    import matplotlib.pyplot as plt
    from src.video_utils import init_video

    projector = SpatioTemporalProjector(seq_id)
    print(f"Projector active for Sequence {seq_str}.")
    
    # Extract the calibration parameters for the requested camera
    if cam_id not in projector.calibrations:
        print(f"Error: Camera {cam_str} does not exist in Sequence {seq_str}.")
        exit()
        
    calib = projector.calibrations[cam_id]
    K = calib['K']
    D = calib['D']

    print(f"Testing Camera {cam_str}.")
    print(f"Resolution: {projector.img_sizes[cam_id]}")
    print(f"FPS: {projector.fps[cam_id]}")
    print(f"timestamp: {projector.timestamp[cam_id]}")
    print(f"Homography:\n{calib['H']}")
    
    # Extract a sample frame
    video_path = os.path.join(AI_CITY_DATA_DIR, "train", seq_str, cam_str, "vdo.avi")
    cap = init_video(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Error: Could not read video file at {video_path}")
    elif K is None or D is None:
        print(f"{cam_str} does not contain distortion.")
    else:
        print(f"Intrinsics:\n{K}")
        print(f"Distortion:\n{D}")

        # Convert BGR (OpenCV default) to RGB for Matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Undistort the entire frame using the intrinsic matrix K and distortion coefficients D
        undistorted_frame = cv2.undistort(frame_rgb, K, D, None, K)
        
        # Plot the original and corrected images side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        axes[0].imshow(frame_rgb)
        axes[0].set_title(f"Original Frame - {seq_str} {cam_str}", fontsize=14)
        axes[0].axis("off")
        
        axes[1].imshow(undistorted_frame)
        axes[1].set_title(f"Undistorted Frame (Radial Correction) - {seq_str} {cam_str}", fontsize=14)
        axes[1].axis("off")
        
        plt.tight_layout()
        
        # Save the visualization to disk
        save_path = f"distortion_test_{seq_str}_{cam_str}.png"
        plt.savefig(save_path)
        print(f"Success! Unit test visualization saved to {save_path}")