"""
Mahalanobis Adapter - Interface for Mahalanobis 3D MOT tracking.

Architecture:
- third_party/mahalanobis_3d_mot/: Original third-party implementation (Argoverse 2 format)
- adapters/mahalanobis_kitti.py: KITTI-compatible adaptation built on third-party code
- This adapter provides clean interface similar to AB3DMOT and MOT3D adapters

The adapter imports and adapts the core tracking logic from the third-party implementation
while providing KITTI-compatible data processing and interfaces.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path
try:
    from .config_manager import PathManager, TrackingConfig
except ImportError:
    from config_manager import PathManager, TrackingConfig


class MahalanobisAdapter:
    """Adapter for Mahalanobis tracking using KITTI-compatible implementation."""
    
    def __init__(self, path_manager=None):
        self.path_manager = path_manager or PathManager()
        self.config = TrackingConfig(self.path_manager)
        # The third-party submodule is Argoverse-specific, so we use our KITTI-compatible implementation
        self.submodule_path = self.path_manager.mahalanobis_submodule_path  # Original third-party (AV2 format)
        self.custom_path = self.path_manager.mahalanobis_custom_path  # Our KITTI-compatible adaptation
        
    def setup_environment(self):
        """Setup environment to use KITTI-compatible implementation."""
        # Add custom path to Python path for imports
        if str(self.custom_path) not in sys.path:
            sys.path.insert(0, str(self.custom_path))
    
    def run_kitti_tracking(self, **kwargs):
        """Run Mahalanobis KITTI tracking using adapter-based implementation.
        
        This method uses the KITTI-compatible implementation in adapters/mahalanobis_kitti.py
        which adapts the core logic from third_party/mahalanobis_3d_mot.
        """
        self.setup_environment()
        
        # Use our KITTI-compatible adapter implementation
        tracking_script = Path(__file__).parent / "mahalanobis_kitti.py"
        
        if not tracking_script.exists():
            print(f"Error: KITTI-compatible Mahalanobis adapter not found at {tracking_script}")
            return False
        
        # Build command with all required arguments
        cmd = ["python3", str(tracking_script)]
        
        # Map adapter arguments to script arguments
        arg_mapping = {
            'obj_type': '--obj_type',
            'det_root': '--det_root', 
            'save_root': '--save_root',
            'match_threshold': '--match_threshold',
            'conf_threshold': '--conf_threshold',
            'use_angular_velocity': '--use_angular_velocity'
        }
        
        # Add arguments to command
        for key, value in kwargs.items():
            if value is not None and key in arg_mapping:
                cmd.extend([arg_mapping[key], str(value)])
        
        print(f"Running: {' '.join(cmd)}")
        
        # Execute the tracking script
        try:
            result = subprocess.run(cmd, cwd=str(tracking_script.parent))
            return result.returncode == 0
        except Exception as e:
            print(f"Error running Mahalanobis tracker: {e}")
            return False
    
    def get_default_config_args(self):
        """Get default configuration arguments for Mahalanobis."""
        return self.config.get_mahalanobis_args()


def main():
    """Command line interface for Mahalanobis adapter."""
    parser = argparse.ArgumentParser(description='Mahalanobis Adapter')
    parser.add_argument('--obj_type', type=str, required=True,
                       choices=['Car', 'Pedestrian', 'Cyclist'],
                       help='Object type')
    parser.add_argument('--det_root', type=str, required=True,
                       help='Detection results root path')
    parser.add_argument('--save_root', type=str, required=True,
                       help='Save results root path')
    parser.add_argument('--ego_info_root', type=str,
                       help='Ego info root path')
    parser.add_argument('--calib_root', type=str,
                       help='Calibration root path')
    parser.add_argument('--covariance_id', type=str, default='1',
                       help='Covariance ID')
    parser.add_argument('--match_distance', type=str, default='iou',
                       help='Match distance metric')
    parser.add_argument('--match_threshold', type=str, default='0.1',
                       help='Match threshold')
    parser.add_argument('--conf_threshold', type=str, default='0.5',
                       help='Confidence threshold')
    parser.add_argument('--match_algorithm', type=str, default='h',
                       help='Match algorithm')
    parser.add_argument('--use_angular_velocity', type=str, default='False',
                       help='Use angular velocity')
    
    args = parser.parse_args()
    
    adapter = MahalanobisAdapter()
    
    # Convert args to kwargs
    kwargs = {k: v for k, v in vars(args).items() if v is not None}
    
    success = adapter.run_kitti_tracking(**kwargs)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()