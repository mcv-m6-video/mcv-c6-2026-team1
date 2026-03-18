import numpy as np
from scipy.spatial.distance import euclidean
from scipy.optimize import linear_sum_assignment

class CityScaleTracker:
    def __init__(self, visual_only=False, visual_threshold=0.7, spatial_threshold=float("inf")):
        self.visual_only = visual_only
        self.visual_threshold = visual_threshold
        self.spatial_threshold = spatial_threshold

    def _compute_distance_matrix(self, tracks_a, tracks_b):
        """Creates a cost matrix between tracklets."""

        def _calculate_track_cost(track_a, track_b):
            # Filter tracks depending on their visual similarity
            visual_dist = euclidean(track_a["features"], track_b["features"])
            if visual_dist > self.visual_threshold:
                return float("inf")
            
            if self.visual_only:
                return visual_dist

            def _extract_trajectory(track):
                times = np.array([p["time"] for p in track["trajectory"]])
                coords = np.array([p["gps"] for p in track["trajectory"]])
                return times, coords
            
            times_a, coords_a = _extract_trajectory(track_a)
            times_b, coords_b = _extract_trajectory(track_b)
            dt_matrix = np.abs(times_a[:, np.newaxis] - times_b[np.newaxis, :])
            closest_b_indices = np.argmin(dt_matrix, axis=1)
            sync_mask = np.min(dt_matrix, axis=1) <= 1.0

            if np.any(sync_mask):
                # Tracks are synchronized by at least 1s
                valid_a_indices = np.where(sync_mask)[0]
                valid_b_indices = closest_b_indices[sync_mask]

                # Get synchronized GPS coordinates
                sync_coords_a = coords_a[valid_a_indices]
                sync_coords_b = coords_b[valid_b_indices]

                # Get mean L2 norm
                dist = float(np.mean(np.linalg.norm(sync_coords_a - sync_coords_b, axis=1)))

                # Apply distance threshold. Account for similar vehicles
                if dist > self.spatial_threshold:
                    return float("inf")
            
            else:
                # No time overlap: verify by maximum velocity
                min_dt_idx = np.argmin(dt_matrix)
                i, j = np.unravel_index(min_dt_idx, dt_matrix.shape)

                speed = np.linalg.norm(coords_a[i] - coords_b[j]) / dt_matrix[i, j]
                max_speed_b = np.max(np.linalg.norm(coords_b[1:] - coords_b[:-1], axis=1) / np.diff(times_b))

                # Apply speed threshold. Account for similar vehicles and GPS noise (allow up to 2x speed)
                if speed > 2*max_speed_b:
                    return float("inf") 
                
            # Return visual cost after spatiotemporal filtering
            return visual_dist

        cost_matrix = np.full((len(tracks_a), len(tracks_b)), float('inf'))
        for i, track_a in enumerate(tracks_a.values()):
            for j, track_b in enumerate(tracks_b.values()):
                cost_matrix[i, j] = _calculate_track_cost(track_a, track_b)
        return cost_matrix

    def associate_tracks(self, global_tracks, local_tracks):
        """Uses the Hungarian algorithm to associate global to local tracks."""
        cost_matrix = self._compute_distance_matrix(global_tracks, local_tracks)
        
        # Match IDs (replace infinite values for the solver)
        row_ind, col_ind = linear_sum_assignment(np.where(np.isinf(cost_matrix), 1e5, cost_matrix))

        global_ids = list(global_tracks.keys())
        local_ids = list(local_tracks.keys())

        matches = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] != float("inf"): # Keep only matches passing the filters
                matches.append((global_ids[r], local_ids[c]))
        return matches
    
    def merge_tracks(self, global_track, local_track):
        merged = {}
        merged['trajectory'] = sorted(global_track['trajectory'] + local_track['trajectory'], key=lambda p: p['time'])

        # Merge features accounting for amount of merges
        n_cams = global_track["n_cams"]
        merged["n_cams"] = n_cams + 1
        mean_features = (global_track["features"]*n_cams + local_track["features"]) / merged["n_cams"]
        merged["features"] = mean_features / np.linalg.norm(mean_features) # Re-normalize after merging
        
        return merged