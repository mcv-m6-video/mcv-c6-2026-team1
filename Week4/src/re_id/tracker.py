import numpy as np
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment

# NOTE: THIS CAN WORK CORRECT, BUT IT WOULD BE BETTER TO ACCOUNT FOR ALL THE FRAMES FOR TRAJECTORY!
#       THIS WOULD CORRECTLY IDENTIFY 2 CARS THAT ARE SIMILAR

# TODO: USE SPATIOTEMPORAL DISTANCE + VISUAL EMBEDDINGS JUST FOR A CHECK FOR THEM TO BE COMPATIBLE!

class CityScaleTracker:
    def __init__(self, projector, box_filter, max_threshold=0.6):
        self.projector = projector
        self.box_filter = box_filter
        self.max_threshold = max_threshold

    def _compute_distance_matrix(self, tracks_a, tracks_b):
        """Creates a cost matrix between tracklets."""
        cost_matrix = np.full((len(tracks_a), len(tracks_b)), float('inf'))
        for i, track_a in enumerate(tracks_a.values()):
            for j, track_b in enumerate(tracks_b.values()):
                cost_matrix[i, j] = self._calculate_pairwise_cost(track_a, track_b)

        return cost_matrix

    def _calculate_pairwise_cost(self, track_a, track_b):
        # 1. Spatio-Temporal Calculation
        gps_a = self.projector.get_ground_plane_coord(track_a['cam_id'], track_a['bbox'])
        gps_b = self.projector.get_ground_plane_coord(track_b['cam_id'], track_b['bbox'])
        
        time_a = self.projector.get_global_time(track_a['cam_id'], track_a['frame'])
        time_b = self.projector.get_global_time(track_b['cam_id'], track_b['frame'])
        
        dist_pixels = np.linalg.norm(gps_a - gps_b)
        time_diff = abs(time_a - time_b)

        # Filters impossible physical movements
        if time_diff > 0:
            speed = dist_pixels / time_diff

            # TODO: INCLUDE ALL THE FRAME INFORMATION!
        else:
            if dist_pixels > 5.0: # Prevents a vehicle from occupying two places at once
                return float('inf')

        # Normalizes the spatio-temporal distance (maps meters to a 0.0 - 1.0 scale)
        st_cost = min(dist_pixels / 1000.0, 1.0) 

        # 2. Box-Grained Visual Calculation
        valid_a = self.box_filter.is_trustworthy(track_a['bbox'])
        valid_b = self.box_filter.is_trustworthy(track_b['bbox'])

        if valid_a and valid_b:
            # Trusts the TransReID feature entirely if both boxes are high quality
            visual_cost = cosine(track_a['features'], track_b['features'])
            final_cost = (0.7 * visual_cost) + (0.3 * st_cost)
        else:
            # Discards the visual feature and relies on physics if truncation occurs
            final_cost = st_cost 
            
        return final_cost

    def associate_tracks(self, global_tracks, local_tracks):
        """Uses the Hungarian algorithm to associate global to local tracks."""
        cost_matrix = self._compute_distance_matrix(global_tracks, local_tracks)
        
        # Replaces infinite values with a high constant for the assignment solver
        solve_matrix = np.where(np.isinf(cost_matrix), 1e5, cost_matrix)
        
        # Match IDs using 
        row_ind, col_ind = linear_sum_assignment(solve_matrix)

        global_ids = list(global_tracks.keys())
        local_ids = list(local_tracks.keys())

        matches = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= self.max_threshold:
                matches.append((global_ids[r], local_ids[c]))
                
        return matches