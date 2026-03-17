import numpy as np
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment

class CityScaleTracker:
    def __init__(self, visual_threshold=0.2, distance_threshold=35.0, speed_threshold=50.0):
        self.visual_threshold = visual_threshold
        self.distance_threshold = distance_threshold
        self.speed_threshold = speed_threshold

    def _compute_distance_matrix(self, tracks_a, tracks_b):
        """Creates a cost matrix between tracklets."""

        def _calculate_track_cost(track_a, track_b):
            # Filter match depending on visual similarity
            visual_dist = cosine(track_a["features"], track_b["features"])
            if visual_dist > self.visual_threshold:
                return float("inf")

            def _extract_trajectory(track):
                times = np.array([p["time"] for p in track["trajectory"]])
                coords = np.array([p["gps"] for p in track["trajectory"]])
                return times, coords
            
            times_a, coords_a = _extract_trajectory(track_a)
            times_b, coords_b = _extract_trajectory(track_b)

            # Find registered instants synchronized by at least 0.5s
            dt_matrix = np.abs(times_a[:, np.newaxis] - times_b[np.newaxis, :])
            closest_b_indices = np.argmin(dt_matrix, axis=1)
            sync_mask = np.min(dt_matrix, axis=1) <= 0.5
            if np.any(sync_mask):
                # Isolate valid pairs
                valid_a_indices = np.where(sync_mask)[0]
                valid_b_indices = closest_b_indices[sync_mask]

                # Get GPS coordinates
                sync_coords_a = coords_a[valid_a_indices]
                sync_coords_b = coords_b[valid_b_indices]

                # Get mean L2 norm for synchronized pairs
                dist = float(np.mean(np.linalg.norm(sync_coords_a - sync_coords_b, axis=1)))

                # Apply distance threshold. Account for similar vehicles
                return float("inf") if dist > self.distance_threshold else dist

            # No time overlap: find closest pair of frames (in time)
            min_dt_idx = np.argmin(dt_matrix)
            i, j = np.unravel_index(min_dt_idx, dt_matrix.shape)

            # Compute speed needed to cross the shortest temporal gap
            dist = np.linalg.norm(coords_a[i] - coords_b[j])
            speed = dist / dt_matrix[i, j]

            # Speed filtering. Account for similar vehicles
            return float("inf") if speed > self.speed_threshold else dist
        

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
        merged["features"] = (global_track["features"]*n_cams + local_track["features"]) / merged["n_cams"]
        
        return merged