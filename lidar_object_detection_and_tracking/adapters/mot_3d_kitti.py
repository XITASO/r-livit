"""
MOT3D KITTI Adapter - Interface for MOT3D tracking on KITTI dataset.

This adapter provides a clean interface to use the third-party MOT3D implementation
for KITTI dataset tracking, following the same pattern as other adapters in this project.

Architecture:
- third_party/mot_3d/: Original third-party implementation (nuScenes/Waymo format)
- This adapter provides KITTI-compatible data processing and interfaces
- Keeps third-party code clean and unmodified
"""

import os
import sys
import math
import numpy as np
import yaml
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add third-party mot_3d to path
current_dir = Path(__file__).parent
third_party_mot3d = current_dir.parent / "third_party" / "mot_3d"
if str(third_party_mot3d) not in sys.path:
    sys.path.insert(0, str(third_party_mot3d))

# Import mot_3d modules
import mot_3d.visualization as visualization
import mot_3d.utils as utils
from mot_3d.data_protos import BBox, Validity
from mot_3d.mot import MOTModel
from mot_3d.frame_data import FrameData

# Fix numpy compatibility issues
import numpy as np
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float

# Add bbox2ego method to BBox class (monkey patch)
def bbox2ego(ego_matrix, box):
    """Convert world bbox to ego coordinate system (inverse of bbox2world)"""
    # Compute inverse transformation matrix
    inv_ego_matrix = np.linalg.inv(ego_matrix)
    
    # center and corners
    corners = np.array(BBox.box2corners2d(box))
    center = BBox.bbox2array(box)[:3][np.newaxis, :]
    center = BBox.box_pts2world(inv_ego_matrix, center)[0]
    corners = BBox.box_pts2world(inv_ego_matrix, corners)
    # heading
    edge_mid_point = (corners[0] + corners[1]) / 2
    yaw = BBox.edge2yaw(center[:2], edge_mid_point[:2])
    
    result = BBox()
    BBox.copy_bbox(result, box)
    result.x, result.y, result.z = center
    result.o = yaw
    return result

# Monkey patch the BBox class
BBox.bbox2ego = staticmethod(bbox2ego)

try:
    from .config_manager import PathManager, TrackingConfig
except ImportError:
    from config_manager import PathManager, TrackingConfig


class KittiLoader:
    """KITTI data loader compatible with MOT3D framework."""
    
    def __init__(self, configs: Dict[str, Any], type_tokens: List[int], 
                 segment_name: str, data_folder: str, det_data_folder: str, 
                 start_frame: int = 0):
        self.configs = configs
        self.segment = segment_name
        self.data_loader = data_folder
        self.det_data_folder = det_data_folder
        self.type_token = type_tokens[0] if type_tokens else 1
        self.start_frame = start_frame
        self.cur_frame = start_frame
        
        # NMS configuration
        self.nms = configs['data_loader']['nms']
        self.nms_thres = configs['data_loader']['nms_thres']
        
        # Load ego info
        self.ego_info = self._load_ego_info(data_folder, segment_name)
        
        # Load detections
        det_file = os.path.join(det_data_folder, f'{segment_name}.txt')
        self.dets = self._load_detection(det_file, self.type_token)
        self.det_type_filter = True
        
        # Point cloud configuration
        self.use_pc = configs['data_loader']['pc']
        if self.use_pc:
            pcs_folder = os.path.join(data_folder, 'point_cloud', segment_name)
            self.pcs = {}
            if os.path.exists(pcs_folder):
                for root, dirs, files in os.walk(pcs_folder):
                    for file in files:
                        if file.endswith('.bin'):
                            pcs = np.fromfile(os.path.join(root, file), dtype=np.float32).reshape(-1, 4)
                            self.pcs[file.split('.')[0]] = pcs
        
        # Frame information
        if isinstance(self.dets, list):  
            self.dets = np.array(self.dets)

        if self.dets.size > 0:  
            self.max_frame = int(np.max(self.dets[:, 0])) + 1
        else:
            self.max_frame = 0
        
    def _load_ego_info(self, data_folder: str, segment_name: str) -> Dict[str, np.ndarray]:
        """Load ego motion information."""
        ego_info_path = os.path.join(data_folder, 'ego_info', f'{segment_name}.npz')
        if not os.path.exists(ego_info_path):
            return {}
            
        ego_info_tmp = np.load(ego_info_path, allow_pickle=True)
        ego_info_tmp = ego_info_tmp['pose_data']
        ego_info = dict()
        for i in range(ego_info_tmp.shape[0]):
            ego_info[str(i)] = ego_info_tmp[i]
        return ego_info
    
    def _load_detection(self, file: str, cat: int) -> np.ndarray:
        """Load detections from file in KITTI format."""
        type_str2id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3}
        
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                list_data = []
                with open(file, 'r') as read_f:
                    list_lines = read_f.readlines()
                    if len(list_lines) == 0:
                        return np.array([])
                    for line in list_lines:
                        line = line.strip('\n').split(' ')
                        if len(line) > 1 and line[1] in type_str2id and type_str2id[line[1]] == cat:
                            line[1] = type_str2id[line[1]]
                            line = [float(i) for i in line]
                            list_data.append(np.array(line))
                dets = np.array(list_data)

            if len(dets.shape) == 1:
                dets = np.expand_dims(dets, axis=0)
            if dets.shape[1] == 0:  # if no detection in a sequence
                return np.array([])
            else:
                return dets
        except Exception as e:
            print(f'Current sequence detection file does not exist: {e}')
            return np.array([])
    
    def __len__(self):
        return self.max_frame
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.cur_frame >= self.max_frame:
            raise StopIteration
        
        cur_det = [self.dets[i, :] for i in range(len(self.dets)) if self.dets[i, 0] == self.cur_frame]
        cur_det = np.array(cur_det)
        if cur_det.size == 0:
            self.cur_frame += 1
            return None
        if cur_det.ndim == 1:
            cur_det = np.expand_dims(cur_det, axis=0)
        
        result = dict()
        result['time_stamp'] = self.cur_frame * 1e-1
        result['ego'] = self.ego_info[str(self.cur_frame)]
        bboxes = np.zeros((cur_det.shape[0], 8))
        # l, w, h
        bboxes[:, 4] = cur_det[:, 12]
        bboxes[:, 5] = cur_det[:, 11]
        bboxes[:, 6] = cur_det[:, 10]

        # x, y, z, o
        bboxes[:, 0:4] = cur_det[:, 17:21]
        # s
        bboxes[:, 7] = cur_det[:, 22]
        
        inst_types = cur_det[:, 1]
        selected_dets = [bboxes[i] for i in range(len(bboxes)) if inst_types[i] in [self.type_token]]
        result['det_types'] = [inst_types[i] for i in range(len(bboxes)) if inst_types[i] in [self.type_token]]
        # BBox: x, y, z, o, l, w, h, s
        result['dets'] = [BBox.bbox2world(result['ego'], BBox.array2bbox(b))
            for b in selected_dets]

        result['pc'] = None
        if self.use_pc:
            pc_key = f'{self.cur_frame:06d}'
            if pc_key in self.pcs:
                pc = self.pcs[pc_key]
                result['pc'] = utils.pc2world(result['ego'], pc)
        
        result['aux_info'] = {'is_key_frame': True}
        result['aux_info']['velos'] = None
        
        if self.nms:
            result['dets'], result['det_types'], result['aux_info']['velos'] = \
                self._frame_nms(result['dets'], result['det_types'], result['aux_info']['velos'], self.nms_thres)
        result['dets'] = [BBox.bbox2array(d) for d in result['dets']]

        self.cur_frame += 1
        return result
    
    def _frame_nms(self, dets, det_types, velos, thres):
        """Apply NMS to detections."""
        from mot_3d.preprocessing import nms
        frame_indexes, frame_types = nms(dets, det_types, thres)
        result_dets = [dets[i] for i in frame_indexes]
        result_velos = None
        if velos is not None:
            result_velos = [velos[i] for i in frame_indexes]
        return result_dets, frame_types, result_velos
    


class MOT3DAdapter:
    """Adapter for MOT3D tracking using third-party implementation."""
    
    def __init__(self, path_manager=None):
        self.path_manager = path_manager or PathManager()
        self.config = TrackingConfig(self.path_manager)
        self.submodule_path = self.path_manager.mot3d_submodule_path
        
    def setup_environment(self):
        """Setup environment to use third-party MOT3D."""
        if str(self.submodule_path) not in sys.path:
            sys.path.insert(0, str(self.submodule_path))
    
    def run_kitti_tracking(self, obj_type: str, data_folder: str, det_data_folder: str,
                          result_folder: str, config_path: str = None, 
                          gt_folder: str = None, start_frame: int = 0, 
                          visualize: bool = False, process: int = 1) -> bool:
        """Run MOT3D tracking on KITTI dataset."""
        self.setup_environment()
        
        # Load configuration
        if config_path is None:
            config_path = self.path_manager.get_mot3d_config_path()
        
        if not os.path.exists(config_path):
            print(f"Error: Config file not found at {config_path}")
            return False
            
        with open(config_path, 'r') as f:
            configs = yaml.load(f, Loader=yaml.Loader)
        
        # Determine type token
        type_token = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3}.get(obj_type, 1)
        
        # Get sequence names
        ego_info_folder = os.path.join(data_folder, 'ego_info')
        if not os.path.exists(ego_info_folder):
            print(f"Error: Ego info folder not found at {ego_info_folder}")
            return False
            
        file_names = sorted(os.listdir(ego_info_folder))
        
        # Process only the first sequence for testing
        if file_names:
            file_name = file_names[0]
            print(f'START TYPE {obj_type} SEQ 1 / 1 (Testing only first sequence)')
            segment_name = file_name.split('.')[0]
            
            # Create data loader
            data_loader = KittiLoader(
                configs, [type_token], segment_name, data_folder, 
                det_data_folder, start_frame
            )
            
            # Load ground truth if available
            gt_bboxes, gt_ids = None, None
            if gt_folder:
                gt_bboxes, gt_ids = self._load_gt_bboxes(gt_folder, data_folder, segment_name, type_token)
            
            # Run tracking
            success = self._sequence_mot(
                configs, data_loader, 0, gt_bboxes, gt_ids, 
                visualize, result_folder, segment_name, data_folder
            )
            
            if not success:
                print(f"Error processing sequence {segment_name}")
                return False
        else:
            print("No sequences found")
            return False
                
        return True
    
    def _load_gt_bboxes(self, gt_folder: str, data_folder: str, 
                       segment_name: str, type_token: int) -> Tuple[List, List]:
        """Load ground truth bounding boxes."""
        # This is a simplified version - can be expanded based on needs
        return [], []
    
    def _sequence_mot(self, configs: Dict[str, Any], data_loader: KittiLoader, 
                     sequence_id: int, gt_bboxes: List, gt_ids: List,
                     visualize: bool, result_folder: str, segment_name: str,
                     data_folder: str) -> bool:
        """Run MOT on a single sequence."""
        try:
            tracker = MOTModel(configs)
            frame_num = len(data_loader)
            IDs, bboxes, local_bboxes, states, types, frames = [], [], [], [], [], []
            
            for frame_index in range(data_loader.cur_frame, frame_num):
                print(f'TYPE {data_loader.type_token} SEQ {sequence_id + 1} Frame {frame_index + 1} / {frame_num}')
                
                # Load frame data
                frame_data = next(data_loader)
                if frame_data is None:
                    continue
                    
                # Create FrameData object
                frame_data = FrameData(
                    dets=frame_data['dets'], 
                    ego=frame_data['ego'], 
                    pc=frame_data['pc'],
                    det_types=frame_data['det_types'], 
                    aux_info=frame_data['aux_info'], 
                    time_stamp=frame_data['time_stamp']
                )
                
                # Run MOT
                results = tracker.frame_mot(frame_data)
                result_pred_bboxes = [trk[0] for trk in results]
                result_pred_ids = [trk[1] for trk in results]
                result_pred_states = [trk[2] for trk in results]
                result_types = [trk[3] for trk in results]
                result_frames = [frame_index for trk in results]
                
                # Visualization (optional)
                if visualize:
                    self._frame_visualization(
                        result_pred_bboxes, result_pred_ids, result_pred_states,
                        gt_bboxes[frame_index] if gt_bboxes and frame_index < len(gt_bboxes) else None,
                        gt_ids[frame_index] if gt_ids and frame_index < len(gt_ids) else None,
                        frame_data.pc, dets=frame_data.dets,
                        name=f'mot3d_{sequence_id}_{frame_index}'
                    )
                
                # Store results
                IDs.append(result_pred_ids)
                result_pred_bboxes = [BBox.bbox2array(bbox) for bbox in result_pred_bboxes]
                result_pred_bboxes_local_tmp = [BBox.bbox2ego(frame_data.ego, BBox.array2bbox(b)) for b in result_pred_bboxes]
                result_pred_bboxes_local = [BBox.bbox2array(bbox) for bbox in result_pred_bboxes_local_tmp]
                bboxes.append(result_pred_bboxes)
                local_bboxes.append(result_pred_bboxes_local)
                states.append(result_pred_states)
                types.append(result_types)
                frames.append(result_frames)
                
            
            # Save results
            self._save_results(IDs, local_bboxes, types, frames, segment_name, 
                             result_folder, data_folder)
            
            return True
            
        except Exception as e:
            print(f"Error in sequence MOT: {e}")
            return False
    
    def _frame_visualization(self, bboxes, ids, states, gt_bboxes=None, gt_ids=None, 
                           pc=None, dets=None, name=''):
        """Visualize tracking results for a frame."""
        try:
            visualizer = visualization.Visualizer2D(name=name, figsize=(12, 12))
            if pc is not None:
                visualizer.handler_pc(pc)
            if gt_bboxes:
                for bbox in gt_bboxes:
                    visualizer.handler_box(bbox, message='', color='black')
            if dets:
                dets = [d for d in dets if d.s >= 0.1]
                for det in dets:
                    visualizer.handler_box(det, message='%.2f' % det.s, color='purple', linestyle='dashed')
            for bbox, id, state in zip(bboxes, ids, states):
                if Validity.valid(state):
                    visualizer.handler_box(bbox, message=str(id), color='red')
                else:
                    visualizer.handler_box(bbox, message=str(id), color='light_blue')
            visualizer.save(f'imgs/{name}.png')
            visualizer.close()
        except Exception as e:
            print(f"Visualization error: {e}")
    
    def _save_results(self, ids, bboxes, types, frames, segment_name, result_folder, data_folder):
        """Save tracking results in KITTI format."""
        os.makedirs(result_folder, exist_ok=True)
        
        result_file = os.path.join(result_folder, f'{segment_name}.txt')
        calib_path = os.path.join(data_folder, 'calib', f'{segment_name}.txt')
        
        # Load calibration
        calib = self._get_calib_from_file(calib_path)
        
        with open(result_file, 'w') as f:
            for i in range(len(ids)):
                self._save_frame_results(ids[i], bboxes[i], types[i], frames[i], calib, f)
    
    def _get_calib_from_file(self, calib_file: str) -> Dict[str, np.ndarray]:
        """Load calibration parameters from KITTI format file."""
        with open(calib_file) as f:
            lines = f.readlines()

        obj = lines[2].strip().split(' ')[1:]
        P2 = np.array(obj, dtype=np.float32)
        obj = lines[3].strip().split(' ')[1:]
        P3 = np.array(obj, dtype=np.float32)
        obj = lines[4].strip().split(' ')[1:]
        R0 = np.array(obj, dtype=np.float32)
        obj = lines[5].strip().split(' ')[1:]
        Tr_velo_to_cam = np.array(obj, dtype=np.float32)

        return {
            'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)
        }
    
    def _save_frame_results(self, ids, bboxes, types, frames, calib, eval_file):
        """Save results for a single frame in KITTI format."""
        type_id2str = {1: 'Car', 2: 'Pedestrian', 3: 'Cyclist'}
        
        for i in range(len(bboxes)):
            if len(bboxes[i]) < 7:
                continue
                
            # bboxes format [x, y, z, rotation_z, l, w, h, score]
            bbox = bboxes[i]
            if len(bbox) == 8:
                x, y, z, rotation_z, l, w, h, score = bbox
            else:
                x, y, z, rotation_z, l, w, h = bbox[:7]
                score = 1.0
            
            type_str = type_id2str.get(types[i], 'Car')
            obj_id = int(ids[i])
            frame = int(frames[i]) if isinstance(frames[i], (int, float, str)) else 0
            
            # Use default values for unmodified fields (as in reference implementation)
            truncated = 0.0
            occluded = 0.0
            alpha = 0.0
            bbox_2d = [0.0, 0.0, 100.0, 100.0]  # Default 2D bbox
            h, w, l = bbox[4], bbox[5], bbox[6]  # Use tracking dimensions
            camera_x, camera_y, camera_z = x, y, z  # Use tracking position
            camera_rotation = rotation_z
            lidar_x, lidar_y, lidar_z = x, y, z
            lidar_rotation = rotation_z
            confidence = score
            placeholder1 = -1
            placeholder2 = -1
            placeholder3 = -1
            
            # Write KITTI tracking format: frame_id, type, tracking_id, truncated, occluded, alpha,
            # 2d_bbox[4], dimensions[3], location_in_camera[3], rotation_in_camera,
            # location_in_lidar[3], rotation_in_lidar, -1, confidence, -1, -1
            line = f"{frame} {type_str} {obj_id} {truncated} {occluded} {alpha} {bbox_2d[0]:.2f} {bbox_2d[1]:.2f} {bbox_2d[2]:.2f} {bbox_2d[3]:.2f} {h:.2f} {w:.2f} {l:.2f} {camera_x:.2f} {camera_y:.2f} {camera_z:.2f} {camera_rotation:.2f} {lidar_x:.2f} {lidar_y:.2f} {lidar_z:.2f} {lidar_rotation:.2f} -1 {confidence:.2f} {placeholder2} {placeholder3}\n"
            eval_file.write(line)


def main():
    """Command line interface for MOT3D KITTI adapter."""
    parser = argparse.ArgumentParser(description='MOT3D KITTI Adapter')
    parser.add_argument('--obj_type', type=str, default='Car', 
                       choices=['Car', 'Pedestrian', 'Cyclist'],
                       help='Object type to track')
    parser.add_argument('--data_folder', type=str, required=True,
                       help='KITTI data folder path')
    parser.add_argument('--det_data_folder', type=str, required=True,
                       help='Detection results folder path')
    parser.add_argument('--result_folder', type=str, required=True,
                       help='Output results folder path')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Configuration file path')
    parser.add_argument('--gt_folder', type=str, default=None,
                       help='Ground truth folder path')
    parser.add_argument('--start_frame', type=int, default=0,
                       help='Start frame index')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable visualization')
    parser.add_argument('--process', type=int, default=1,
                       help='Number of processes')
    
    args = parser.parse_args()
    
    adapter = MOT3DAdapter()
    
    success = adapter.run_kitti_tracking(
        obj_type=args.obj_type,
        data_folder=args.data_folder,
        det_data_folder=args.det_data_folder,
        result_folder=args.result_folder,
        config_path=args.config_path,
        gt_folder=args.gt_folder,
        start_frame=args.start_frame,
        visualize=args.visualize,
        process=args.process
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
