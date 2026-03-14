import numpy as np
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment

# TODO: CHECK!
class CityScaleTracker:
    def __init__(self, projector, box_filter, max_speed_kmh=120):
        self.projector = projector
        self.box_filter = box_filter
        self.max_speed_ms = max_speed_kmh / 3.6

    # NOTE: TRACKS ARE NOT DICTS, NOT LISTS!
    def compute_distance_matrix(self, tracks_cam_a, tracks_cam_b):
        """Creates a cost matrix between tracklets from two different cameras."""
        num_a = len(tracks_cam_a)
        num_b = len(tracks_cam_b)
        cost_matrix = np.full((num_a, num_b), float('inf'))

        for i, track_a in enumerate(tracks_cam_a):
            for j, track_b in enumerate(tracks_cam_b):
                cost_matrix[i, j] = self._calculate_pairwise_cost(track_a, track_b)
                
        return cost_matrix

    def _calculate_pairwise_cost(self, track_a, track_b):
        # 1. Spatio-Temporal Calculation
        gps_a = self.projector.get_ground_plane_coord(track_a['cam_id'], track_a['bbox'])
        gps_b = self.projector.get_ground_plane_coord(track_b['cam_id'], track_b['bbox'])
        
        time_a = self.projector.get_global_time(track_a['cam_id'], track_a['frame'])
        time_b = self.projector.get_global_time(track_b['cam_id'], track_b['frame'])
        
        dist_meters = np.linalg.norm(gps_a - gps_b)
        time_diff = abs(time_a - time_b)

        # Filters impossible physical movements
        if time_diff > 0:
            speed = dist_meters / time_diff
            if speed > self.max_speed_ms:
                return float('inf')
        else:
            if dist_meters > 5.0: # Prevents a vehicle from occupying two places at once
                return float('inf')

        # Normalizes the spatio-temporal distance (maps meters to a 0.0 - 1.0 scale)
        st_cost = min(dist_meters / 1000.0, 1.0) 

        # 2. Box-Grained Visual Calculation
        valid_a = self.box_filter.is_trustworthy(track_a['bbox'])
        valid_b = self.box_filter.is_trustworthy(track_b['bbox'])

        if valid_a and valid_b:
            # Trusts the TransReID feature entirely if both boxes are high quality
            visual_cost = cosine(track_a['feature'], track_b['feature'])
            final_cost = (0.7 * visual_cost) + (0.3 * st_cost)
        else:
            # Discards the visual feature and relies on physics if truncation occurs
            final_cost = st_cost 
            
        return final_cost

    def associate_cameras(self, tracks_cam_a, tracks_cam_b):
        """Uses the Hungarian algorithm to assign identical track IDs between cameras."""
        cost_matrix = self.compute_distance_matrix(tracks_cam_a, tracks_cam_b)
        
        # Replaces infinite values with a high constant for the assignment solver
        solve_matrix = np.where(np.isinf(cost_matrix), 1e5, cost_matrix)
        
        row_ind, col_ind = linear_sum_assignment(solve_matrix)
        
        matches = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] != float('inf'):
                matches.append((tracks_cam_a[r]['track_id'], tracks_cam_b[c]['track_id']))
                
        return matches