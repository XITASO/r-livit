"""
MOT3D Adapter - Interface between plugin calls and submodule implementation.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path
try:
    from .config_manager import PathManager, TrackingConfig
    from .mot_3d_kitti import MOT3DAdapter as KittiMOT3DAdapter
except ImportError:
    from config_manager import PathManager, TrackingConfig
    from mot_3d_kitti import MOT3DAdapter as KittiMOT3DAdapter


class MOT3DAdapter:
    """Adapter for MOT3D tracking using submodule implementation."""
    
    def __init__(self, path_manager=None):
        self.path_manager = path_manager or PathManager()
        self.config = TrackingConfig(self.path_manager)
        self.submodule_path = self.path_manager.mot3d_submodule_path
        # Use our KITTI-compatible implementation
        self.kitti_adapter = KittiMOT3DAdapter(path_manager)
        
    def setup_environment(self):
        """Setup environment to use submodule MOT3D."""
        # Add submodule path to Python path
        if str(self.submodule_path) not in sys.path:
            sys.path.insert(0, str(self.submodule_path))
    
    def run_kitti_inference(self, config_path=None, obj_type='Car', data_folder=None, 
                           det_data_folder=None, result_folder=None, gt_folder=None,
                           start_frame=0, visualize=False, process=1, **kwargs):
        """Run MOT3D KITTI inference using our KITTI-compatible implementation."""
        self.setup_environment()
        
        # Use our KITTI adapter implementation
        if config_path is None:
            config_path = self.path_manager.get_mot3d_config_path()
        
        # Validate required arguments
        if not data_folder or not det_data_folder or not result_folder:
            print("Error: data_folder, det_data_folder, and result_folder are required")
            return False
        
        print(f"Running MOT3D KITTI tracking with:")
        print(f"  Object type: {obj_type}")
        print(f"  Data folder: {data_folder}")
        print(f"  Detection folder: {det_data_folder}")
        print(f"  Result folder: {result_folder}")
        print(f"  Config path: {config_path}")
        
        # Create result directory
        os.makedirs(result_folder, exist_ok=True)
        
        # Run tracking using our KITTI adapter
        success = self.kitti_adapter.run_kitti_tracking(
            obj_type=obj_type,
            data_folder=data_folder,
            det_data_folder=det_data_folder,
            result_folder=result_folder,
            config_path=config_path,
            gt_folder=gt_folder,
            start_frame=start_frame,
            visualize=visualize,
            process=process
        )
        
        if success:
            print("✓ MOT3D KITTI tracking completed successfully")
        else:
            print("✗ MOT3D KITTI tracking failed")
            
        return success
    
    def get_default_config_args(self):
        """Get default configuration arguments for MOT3D."""
        return {
            'config_path': str(self.path_manager.get_mot3d_config_path()),
        }


def main():
    """Command line interface for MOT3D adapter."""
    parser = argparse.ArgumentParser(description='MOT3D Adapter')
    parser.add_argument('--config_path', type=str, 
                       help='Config file path')
    parser.add_argument('--obj_type', type=str, default='Car', 
                       choices=['Car', 'Pedestrian', 'Cyclist'],
                       help='Object type')
    parser.add_argument('--det_data_folder', type=str, required=True,
                       help='Detection results folder')
    parser.add_argument('--result_folder', type=str, required=True,
                       help='Results folder')
    parser.add_argument('--data_folder', type=str, required=True,
                       help='Data folder')
    parser.add_argument('--gt_folder', type=str,
                       help='Ground truth folder')
    parser.add_argument('--start_frame', type=int, default=0,
                       help='Start frame index')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable visualization')
    parser.add_argument('--process', type=int, default=1,
                       help='Number of processes')
    parser.add_argument('--name', type=str, default='simpletrack',
                       help='Tracker name (for compatibility)')
    parser.add_argument('--det_name', type=str, default='kitti_pointpillar',
                       help='Detector name (for compatibility)')
    
    args = parser.parse_args()
    
    adapter = MOT3DAdapter()
    
    # Convert args to kwargs, excluding compatibility args
    kwargs = {k: v for k, v in vars(args).items() if v is not None and k not in ['name', 'det_name']}
    
    success = adapter.run_kitti_inference(**kwargs)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()