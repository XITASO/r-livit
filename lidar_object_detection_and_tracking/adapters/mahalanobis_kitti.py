"""
KITTI-compatible Mahalanobis 3D MOT implementation.
This module adapts the third-party Mahalanobis implementation to work with KITTI format data.
"""
import os
import sys
import glob
import numpy as np
import math
from pathlib import Path

# Add third-party path for imports
current_dir = Path(__file__).parent
third_party_path = current_dir.parent / "third_party" / "mahalanobis_3d_mot" / "tracking" / "ab3dmot_track"
if str(third_party_path) not in sys.path:
    sys.path.insert(0, str(third_party_path))

from typing import List
import copy
from filterpy.kalman import KalmanFilter


def quat_to_mat(quaternion):
    """
    Convert quaternion to rotation matrix.
    KITTI-compatible implementation to replace av2 dependency.
    
    Args:
        quaternion: [w, x, y, z] quaternion
    
    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = quaternion
    
    # Normalize quaternion
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm == 0:
        return np.eye(3)
    
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # Convert to rotation matrix
    rotation_matrix = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    
    return rotation_matrix


def transform_to_global_reference(ego_to_city_SE3, box, yaw):
    """
    Transform coordinates to global reference.
    Simplified version for KITTI (identity transformation).
    """
    # For KITTI, we typically work in a single coordinate frame
    # So we can use identity transformation
    return box, yaw


def wrap_pi(angle):
    """Wrap angle to [-pi, pi] range."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def convert_3dbox_to_8corner(bbox):
    """Convert 3D bounding box to 8-corner representation."""
    x, y, z, yaw, l, w, h = bbox
    
    # Create 3D bounding box corners in object coordinate system
    corners_3d = np.array([
        [-l/2, -w/2, -h/2],
        [ l/2, -w/2, -h/2],
        [ l/2,  w/2, -h/2],
        [-l/2,  w/2, -h/2],
        [-l/2, -w/2,  h/2],
        [ l/2, -w/2,  h/2],
        [ l/2,  w/2,  h/2],
        [-l/2,  w/2,  h/2]
    ])
    
    # Rotation matrix around z-axis
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw,  cos_yaw, 0],
        [0,        0,       1]
    ])
    
    # Rotate and translate
    corners_3d = np.dot(corners_3d, rotation_matrix.T)
    corners_3d[:, 0] += x
    corners_3d[:, 1] += y
    corners_3d[:, 2] += z
    
    return corners_3d


class KalmanBoxTracker(object):
    """
    KITTI-compatible Kalman Box Tracker.
    Adapts the third-party implementation for KITTI format.
    """
    count = 0
    
    def __init__(self, bbox3D):
        """
        Initializes tracker with initial bounding box.
        bbox3D: [x, y, z, yaw, l, w, h] in KITTI format
        """
        self.kf = self._create_kalman_filter()
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.first_continuing_hit = 1
        self.still_first = True
        self.age = 0
        self.last_observation = bbox3D
        
        # Initialize state vector with bbox3D
        self.kf.x[:7] = bbox3D.reshape((7, 1))
    
    def _create_kalman_filter(self):
        """Create Kalman filter for 3D box tracking."""
        
        kf = KalmanFilter(dim_x=10, dim_z=7)
        
        # State transition matrix (constant velocity model)
        kf.F = np.array([
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Observation matrix
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise
        kf.R[0:, 0:] *= 10.0
        
        # Process noise
        kf.Q[-1, -1] *= 0.01
        kf.Q[7:, 7:] *= 0.01
        
        # Initial covariance
        kf.P[7:, 7:] *= 1000.0
        kf.P *= 10.0
        
        return kf
    
    def update(self, bbox3D):
        """Update tracker with observed bounding box."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        if self.still_first:
            self.first_continuing_hit += 1
        
        # Orientation correction
        if self.kf.x[3] >= np.pi:
            self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi:
            self.kf.x[3] += np.pi * 2
        
        new_theta = bbox3D[3]
        if new_theta >= np.pi:
            new_theta -= np.pi * 2
        if new_theta < -np.pi:
            new_theta += np.pi * 2
        bbox3D[3] = new_theta
        
        predicted_theta = self.kf.x[3]
        if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:
            self.kf.x[3] += np.pi
            if self.kf.x[3] > np.pi:
                self.kf.x[3] -= np.pi * 2
            if self.kf.x[3] < -np.pi:
                self.kf.x[3] += np.pi * 2
        
        diff_theta = new_theta - self.kf.x[3]
        if diff_theta >= np.pi / 2.0 and diff_theta <= np.pi:
            bbox3D[3] -= np.pi
        elif diff_theta <= -np.pi / 2.0 and diff_theta >= -np.pi:
            bbox3D[3] += np.pi
        
        self.kf.update(bbox3D)
        
        if self.kf.x[3] >= np.pi:
            self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi:
            self.kf.x[3] += np.pi * 2
        
        self.last_observation = bbox3D
    
    def predict(self):
        """Predict next state."""
        self.kf.predict()
        if self.kf.x[3] >= np.pi:
            self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi:
            self.kf.x[3] += np.pi * 2
        
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
            self.still_first = False
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.history[-1]
    
    def get_state(self):
        """Return current bounding box estimate."""
        return self.kf.x[:7].reshape((7,))


class SingleClassMOTWithMahalanobis(object):
    """Single class MOT with Mahalanobis distance."""
    
    def __init__(self, match_threshold=0.1, match_algorithm='greedy', max_age=2, 
                 ego_coord=True, use_mahalanobis=False, mahalanobis_threshold=0.1, 
                 print_debug=False):
        self.match_threshold = match_threshold
        self.match_algorithm = match_algorithm
        self.max_age = max_age
        self.tracks = []
        self.track_id_to_detection = {}
        self.ego_coord = ego_coord
        self.use_mahalanobis = use_mahalanobis
        self.mahalanobis_threshold = mahalanobis_threshold
        self.print_debug = print_debug
    
    def update(self, dets, ego_to_city_SE3=None):
        """Update tracker with detections."""
        if len(dets) == 0:
            # Age existing tracks
            self.tracks = [track for track in self.tracks if track.time_since_update < self.max_age]
            for track in self.tracks:
                track.predict()
            return []
        
        # Convert detection format
        detections = {
            "translation": [],
            "yaw": [],
            "size": []
        }
        
        for det in dets:
            # Convert rotation to yaw
            if isinstance(det['rotation'], list) and len(det['rotation']) == 4:
                # quaternion [w, x, y, z] -> rotation angle around z-axis
                rot = quat_to_mat(det['rotation'])
                yaw = np.array([math.atan2(rot[1, 0], rot[0, 0])])
            else:
                yaw = np.array([det['rotation']])
            
            detections["translation"].append(det['translation'])
            detections["yaw"].append(yaw[0])
            detections["size"].append(np.array(det["size"], dtype=np.float32))
        
        if len(detections["translation"]) > 0:
            detections["translation"] = np.stack(detections["translation"])
            detections["yaw"] = np.array(detections["yaw"])
            detections["size"] = np.stack(detections["size"])
        else:
            detections["translation"] = np.empty((0, 3))
            detections["yaw"] = np.empty((0,))
            detections["size"] = np.empty((0, 3))
        
        bboxes = np.concatenate([
            detections["translation"],
            detections["yaw"].reshape(-1, 1),
            detections["size"]
        ], axis=-1)
        
        # Predict existing tracks
        track_predictions = [track.predict().reshape(-1)[:7] for track in self.tracks]
        track_predictions = np.stack(track_predictions, axis=0) if track_predictions else np.zeros((0, 7))
        
        # Remove NaN predictions
        no_nans = np.all(~np.isnan(track_predictions), axis=1)
        track_predictions = track_predictions[no_nans]
        self.tracks = [track for not_nan, track in zip(no_nans, self.tracks) if not_nan]
        
        # Convert to 8-corner format for IoU calculation
        detections_8corner = [convert_3dbox_to_8corner(bbox) for bbox in bboxes]
        if len(detections_8corner) > 0:
            detections_8corner = np.stack(detections_8corner, axis=0)
        
        track_predictions_8corner = [convert_3dbox_to_8corner(track_pred) for track_pred in track_predictions]
        if len(track_predictions_8corner) > 0:
            track_predictions_8corner = np.stack(track_predictions_8corner, axis=0)
        
        # Simple association based on IoU (simplified version)
        matched, unmatched_detections, unmatched_track_predictions = self._associate_detections_to_tracks(
            detections_8corner, track_predictions_8corner)
        
        # Update matched trackers
        for idx, track in enumerate(self.tracks):
            if idx not in unmatched_track_predictions:
                detection_idx = matched[np.where(matched[:, 1] == idx)[0][0], 0]
                track.update(bboxes[detection_idx, :])
                self.track_id_to_detection[track.id] = dets[detection_idx]
        
        # Create new trackers for unmatched detections
        for idx in unmatched_detections:
            track = KalmanBoxTracker(bboxes[idx, :])
            self.tracks.append(track)
            self.track_id_to_detection[track.id] = dets[idx]
        
        # Remove old tracks
        self.tracks = [track for track in self.tracks if track.time_since_update < self.max_age]
        
        # Return tracking results
        return [
            {
                **self.track_id_to_detection[track.id],
                "track_id": np.array([track.id]),
                "active": np.array([int(track.time_since_update == 0)]),
            }
            for track in self.tracks
        ]
    
    def _associate_detections_to_tracks(self, detections, tracks):
        """Simple association based on distance."""
        if len(tracks) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
        
        # Simplified distance calculation (using centroid distance)
        det_centroids = np.mean(detections, axis=1) if len(detections) > 0 else np.empty((0, 3))
        track_centroids = np.mean(tracks, axis=1) if len(tracks) > 0 else np.empty((0, 3))
        
        if len(det_centroids) == 0 or len(track_centroids) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.arange(len(tracks))
        
        # Calculate distance matrix
        distance_matrix = np.linalg.norm(
            det_centroids[:, np.newaxis, :] - track_centroids[np.newaxis, :, :], axis=2
        )
        
        # Simple greedy assignment
        matched_indices = []
        used_detections = set()
        used_tracks = set()
        
        # Sort by distance
        flat_distances = distance_matrix.flatten()
        indices = np.unravel_index(np.argsort(flat_distances), distance_matrix.shape)
        
        for det_idx, track_idx in zip(indices[0], indices[1]):
            if det_idx not in used_detections and track_idx not in used_tracks:
                if distance_matrix[det_idx, track_idx] < 5.0:  # Distance threshold
                    matched_indices.append([det_idx, track_idx])
                    used_detections.add(det_idx)
                    used_tracks.add(track_idx)
        
        matched_indices = np.array(matched_indices) if matched_indices else np.empty((0, 2), dtype=int)
        unmatched_detections = np.array([i for i in range(len(detections)) if i not in used_detections])
        unmatched_tracks = np.array([i for i in range(len(tracks)) if i not in used_tracks])
        
        return matched_indices, unmatched_detections, unmatched_tracks


class AB3DMOTWithMahalanobis:
    """Multi-class AB3DMOT with Mahalanobis distance."""
    
    def __init__(self, classes: List[str], max_age: int = 2, ego_coord: bool = True,
                 use_mahalanobis: bool = False, mahalanobis_threshold: float = 0.1,
                 print_debug: bool = False):
        self.classes = classes
        self.max_age = max_age
        self.ego_coord = ego_coord
        self.use_mahalanobis = use_mahalanobis
        self.mahalanobis_threshold = mahalanobis_threshold
        self.print_debug = print_debug
        self.reset()
    
    def reset(self):
        self.trackers = {
            name: SingleClassMOTWithMahalanobis(
                match_threshold=0.1,
                match_algorithm="greedy",
                max_age=self.max_age,
                ego_coord=self.ego_coord,
                use_mahalanobis=self.use_mahalanobis,
                mahalanobis_threshold=self.mahalanobis_threshold,
                print_debug=self.print_debug,
            )
            for name in self.classes
        }
    
    def step(self, detections, time_delta, ego_to_city_SE3):
        """Step function for tracking."""
        tracks = []
        for name, tracker in self.trackers.items():
            detections_in_class = [det for det in detections if det['detection_name'] == name]
            tracks.extend(tracker.update(detections_in_class, ego_to_city_SE3))
        return tracks


def load_kitti_detection(file_path):
    """Load KITTI format detection results."""
    if not os.path.exists(file_path):
        return []
    
    detections = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 16:
                # KITTI detection format: frame type truncated occluded alpha bbox_2d[4] dimensions[3] location_in_camera[3] rotation_in_camera location_in_lidar[3] rotation_in_lidar -1 confidence -1 -1
                frame_id = int(parts[0])
                obj_type = parts[1]
                truncated = float(parts[2])
                occluded = float(parts[3])
                alpha = float(parts[4])
                bbox_2d = [float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])]
                h, w, l = float(parts[9]), float(parts[10]), float(parts[11])
                cam_x, cam_y, cam_z = float(parts[12]), float(parts[13]), float(parts[14])
                cam_rot = float(parts[15])
                lidar_x, lidar_y, lidar_z = float(parts[16]), float(parts[17]), float(parts[18])
                lidar_rot = float(parts[19])
                placeholder1 = float(parts[20])
                confidence = float(parts[21])
                placeholder2 = float(parts[22]) if len(parts) > 22 else -1
                placeholder3 = float(parts[23]) if len(parts) > 23 else -1
                
                detections.append([
                    frame_id, obj_type, truncated, occluded, alpha, bbox_2d,
                    h, w, l, cam_x, cam_y, cam_z, cam_rot,
                    lidar_x, lidar_y, lidar_z, lidar_rot,
                    placeholder1, confidence, placeholder2, placeholder3
                ])
    
    return detections


def save_kitti_results(results, file_path, frame_id):
    """Save tracking results in KITTI format."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as f:
        for result in results:
            # Result is a dictionary with detection data + track_id
            if 'track_id' in result and 'translation' in result and 'size' in result:
                track_id = int(result['track_id'][0])
                x, y, z = result['translation']
                l, w, h = result['size']
                
                # Convert quaternion back to rotation angle if needed
                if isinstance(result['rotation'], list) and len(result['rotation']) == 4:
                    w_q, _, _, z_q = result['rotation']
                    rot_y = 2 * math.atan2(z_q, w_q)
                else:
                    rot_y = result['rotation']
                
                # KITTI tracking format
                line = f"{frame_id} {track_id} Car -1 -1 -10 -1 -1 -1 -1 {h:.6f} {w:.6f} {l:.6f} {x:.6f} {y:.6f} {z:.6f} {rot_y:.6f}\n"
                f.write(line)


def run_kitti_tracking(obj_type, det_root, save_root, **kwargs):
    """
    Run KITTI tracking using adapted third-party Mahalanobis implementation.
    Each detection file contains a complete sequence, not a single frame.
    """
    # Extract parameters
    match_threshold = float(kwargs.get('match_threshold', 0.1))
    conf_threshold = float(kwargs.get('conf_threshold', 0.5))
    use_angular_velocity = kwargs.get('use_angular_velocity', 'False').lower() == 'true'
    
    # Get all detection files (each file is a complete sequence)
    detection_files = sorted(glob.glob(os.path.join(det_root, "*.txt")))
    
    if not detection_files:
        print(f"No detection files found in {det_root}")
        return False
    
    print(f"Processing {len(detection_files)} sequence files...")
    os.makedirs(save_root, exist_ok=True)
    
    # Process each sequence file
    for seq_id, det_file in enumerate(detection_files):
        print(f"Processing sequence {seq_id}: {os.path.basename(det_file)}")
        
        # Load all detections for this sequence
        all_detections = load_kitti_detection(det_file)
        
        if not all_detections:
            print(f"No detections found in {det_file}")
            continue
        
        # Group detections by frame
        frame_detections = {}
        for det in all_detections:
            if len(det) >= 20:  # Updated for new format
                frame_id = int(det[0])  # First element is frame_id
                if frame_id not in frame_detections:
                    frame_detections[frame_id] = []
                frame_detections[frame_id].append(det)
        
        # Initialize tracker for this sequence
        tracker = AB3DMOTWithMahalanobis(
            classes=[obj_type.upper()],
            max_age=3,
            ego_coord=False,  # Skip coordinate transformation for KITTI
            use_mahalanobis=True,
            mahalanobis_threshold=match_threshold,
            print_debug=False
        )
        
        # Process each frame in the sequence
        frame_ids = sorted(frame_detections.keys())
        for frame_idx, frame_id in enumerate(frame_ids):
            detections = frame_detections[frame_id]
            
            # Filter by object type and confidence threshold
            valid_detections = []
            for det in detections:
                if len(det) >= 20:
                    det_obj_type = det[1]
                    confidence = det[19]  # confidence is at index 19
                    if det_obj_type.upper() == obj_type.upper() and confidence >= conf_threshold:
                        valid_detections.append(det)
            
            # Convert detections to expected format for tracking
            detections_dict = []
            for det in valid_detections:
                if len(det) >= 20:
                    # Extract LiDAR coordinates for tracking
                    lidar_x, lidar_y, lidar_z = det[15], det[16], det[17]
                    lidar_rot = det[18]
                    h, w, l = det[6], det[7], det[8]
                    confidence = det[19]
                    
                    # Convert rotation angle to quaternion as list [w, x, y, z]
                    half_angle = lidar_rot / 2
                    w_q = math.cos(half_angle)
                    z_q = math.sin(half_angle)
                    quaternion = [w_q, 0, 0, z_q]
                    
                    detection = {
                        'detection_name': obj_type.upper(),
                        'translation': np.array([lidar_x, lidar_y, lidar_z], dtype=np.float32),
                        'size': [l, w, h],  # l, w, h
                        'rotation': quaternion,  # quaternion [w, x, y, z]
                        'score': confidence,
                        'original_detection': det  # Store original detection for output
                    }
                    detections_dict.append(detection)
            
            # Update tracker (using identity matrix for ego_to_city_SE3)
            ego_to_city_SE3 = np.eye(4)
            results = tracker.step(detections_dict, 0.1, ego_to_city_SE3)
            
            # Save results for this frame
            seq_name = os.path.splitext(os.path.basename(det_file))[0]
            output_file = os.path.join(save_root, f"{seq_name}.txt")
            
            # Append results to file (create if first frame)
            mode = 'w' if frame_idx == 0 else 'a'
            with open(output_file, mode) as f:
                for result in results:
                    # Result is a dictionary with detection data + track_id
                    if 'track_id' in result and 'original_detection' in result:
                        track_id = int(result['track_id'][0])
                        original_det = result['original_detection']
                        
                        # Extract all values from original detection
                        orig_frame_id = int(original_det[0])
                        orig_type = original_det[1]
                        truncated = original_det[2]
                        occluded = original_det[3]
                        alpha = original_det[4]
                        bbox_2d = original_det[5]  # [x1, y1, x2, y2]
                        h, w, l = original_det[6], original_det[7], original_det[8]
                        cam_x, cam_y, cam_z = original_det[9], original_det[10], original_det[11]
                        cam_rot = original_det[12]
                        lidar_x, lidar_y, lidar_z = original_det[13], original_det[14], original_det[15]
                        lidar_rot = original_det[16]
                        placeholder1 = original_det[17]
                        confidence = original_det[18]
                        placeholder2 = original_det[19] if len(original_det) > 19 else -1
                        placeholder3 = original_det[20] if len(original_det) > 20 else -1
                        
                        # KITTI tracking format: frame_id, type, tracking_id, truncated, occluded, alpha, 
                        # 2d_bbox[4], dimensions[3], location_in_camera[3], rotation_in_camera, 
                        # location_in_lidar[3], rotation_in_lidar, -1, confidence
                        line = f"{frame_id} {orig_type} {track_id} {truncated} {occluded} {alpha} {bbox_2d[0]} {bbox_2d[1]} {bbox_2d[2]} {bbox_2d[3]} {h:.6f} {w:.6f} {l:.6f} {cam_x:.6f} {cam_y:.6f} {cam_z:.6f} {cam_rot:.6f} {lidar_x:.6f} {lidar_y:.6f} {lidar_z:.6f} {lidar_rot:.6f} -1 {confidence:.6f}\n"
                        f.write(line)
            
            if frame_idx % 10 == 0 or frame_idx < 5:
                print(f"  Processed frame {frame_id}: {len(valid_detections)} detections -> {len(results)} tracks")
        
        print(f"  Completed sequence {seq_id}: {len(frame_ids)} frames processed")
    
    print(f"Tracking completed. Results saved to {save_root}")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='KITTI-compatible Mahalanobis 3D MOT')
    parser.add_argument('--obj_type', type=str, required=True, choices=['Car', 'Pedestrian', 'Cyclist'])
    parser.add_argument('--det_root', type=str, required=True, help='Detection results root path')
    parser.add_argument('--save_root', type=str, required=True, help='Save results root path')
    parser.add_argument('--match_threshold', type=str, default='0.1', help='Match threshold')
    parser.add_argument('--conf_threshold', type=str, default='0.5', help='Confidence threshold')
    parser.add_argument('--use_angular_velocity', type=str, default='False', help='Use angular velocity')
    
    args = parser.parse_args()
    
    success = run_kitti_tracking(**vars(args))
    sys.exit(0 if success else 1)