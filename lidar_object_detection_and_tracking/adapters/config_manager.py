"""
Configuration management for R-LiViT tracking system.
Handles path mapping between plugins and submodules.
"""
import os
from pathlib import Path


class PathManager:
    """Manages paths for different tracking components."""
    
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
    
    @property
    def third_party_dir(self):
        return self.base_dir / "third_party"
    
    @property 
    def ab3dmot_submodule_path(self):
        return self.third_party_dir / "ab3dmot" / "v2x" / "AB3DMOT_plugin"
    
    @property
    def mot3d_submodule_path(self):
        return self.third_party_dir / "mot_3d"
        
    @property
    def mahalanobis_submodule_path(self):
        return self.third_party_dir / "mahalanobis_3d_mot" / "tracking" / "ab3dmot_track"
    
    @property
    def mahalanobis_custom_path(self):
        """Keep custom mahalanobis implementation"""
        return self.base_dir / "mahalanobis_3d_mot_plugin"
    
    def get_ab3dmot_config_path(self, config_name="KITTI.yml"):
        """Get AB3DMOT configuration file path"""
        return self.ab3dmot_submodule_path / "configs" / config_name
    
    def get_mot3d_config_path(self, config_name="mot3d_kitti_config.yaml"):
        """Get MOT3D configuration file path"""
        # Use adapters config directory
        adapters_config = self.base_dir / "adapters" / "configs" / config_name
        if adapters_config.exists():
            return adapters_config
        
        # Fallback to submodule path
        submodule_config = self.mot3d_submodule_path / "configs" / "waymo_configs" / "vc_kf_giou.yaml"
        if submodule_config.exists():
            return submodule_config
        plugin_config = self.base_dir / "mot_3d_plugin" / "configs" / "kitti_configs" / config_name
        return plugin_config


class TrackingConfig:
    """Configuration for tracking parameters."""
    
    def __init__(self, path_manager=None):
        self.path_manager = path_manager or PathManager()
    
    def get_ab3dmot_args(self, input_path, output_path, category="Car"):
        """Get standardized AB3DMOT arguments."""
        return {
            'dataset': 'KITTI',
            'det_name': 'openpcdet', 
            'cat': category,
            'input_path': input_path,
            'output_path': output_path,
            'config_path': str(self.path_manager.get_ab3dmot_config_path()),
            'vis_path': ""
        }
    
    def get_mot3d_args(self, input_path, output_path, category="Car"):
        """Get standardized MOT3D arguments."""
        config_path = self.path_manager.get_mot3d_config_path()
        return {
            'config_path': str(config_path),
            'input_path': input_path,
            'output_path': output_path,
            'category': category
        }
    
    def get_mahalanobis_args(self, **kwargs):
        """Get Mahalanobis tracker arguments."""
        defaults = {
            'covariance_id': '1',
            'match_distance': 'iou', 
            'match_threshold': '0.1',
            'conf_threshold': '0.5',
            'match_algorithm': 'h',
            'use_angular_velocity': False
        }
        defaults.update(kwargs)
        return defaults