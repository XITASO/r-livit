"""
AB3DMOT Adapter - Interface between plugin calls and submodule implementation.
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


class AB3DMOTAdapter:
    """Adapter for AB3DMOT tracking using submodule implementation."""
    
    def __init__(self, path_manager=None):
        self.path_manager = path_manager or PathManager()
        self.config = TrackingConfig(self.path_manager)
        self.submodule_path = self.path_manager.ab3dmot_submodule_path
        
    def setup_environment(self):
        """Setup environment to use submodule AB3DMOT."""
        # Add submodule path to Python path
        if str(self.submodule_path) not in sys.path:
            sys.path.insert(0, str(self.submodule_path))
    
    def run_main_tracking(self, dataset="KITTI", det_name="openpcdet", cat="Car",
                         input_path="", output_path="", config_path=None, vis_path=""):
        """Run AB3DMOT main tracking script using submodule or fallback to plugin."""
        self.setup_environment()
        
        # Try submodule first, fallback to plugin if submodule has issues
        main_script = self.submodule_path / "main_tracking.py"
        plugin_script = self.submodule_path / "main_tracking.py"  # Use the same path as submodule
        
        if config_path is None:
            # Try submodule config first
            try:
                config_path = self.path_manager.get_ab3dmot_config_path()
                if not config_path.exists():
                    config_path = self.submodule_path / "configs" / "KITTI.yml"
            except:
                config_path = self.submodule_path / "configs" / "KITTI.yml"
        
        # Use submodule implementation
        if not main_script.exists():
            print("Error: AB3DMOT main script not found!")
            return False
        else:
            print("Using AB3DMOT submodule implementation...")
            script_dir = self.submodule_path
        
        cmd = [
            "python3", str(main_script),
            "--dataset", dataset,
            "--det_name", det_name, 
            "--cat", cat,
            "--input-path", input_path,
            "--output-path", output_path,
            "--config-path", str(config_path)
        ]
        
        if vis_path:
            cmd.extend(["--vis-path", vis_path])
        
        # Change to script directory for relative imports
        original_cwd = os.getcwd()
        try:
            os.chdir(script_dir)
            result = subprocess.run(cmd, cwd=str(script_dir))
            return result.returncode == 0
        finally:
            os.chdir(original_cwd)
    
    def run_eval_tracking(self, track_eval_gt_path, calib_gt_path, track_results_path, 
                         track_eval_output_path, cat="Car", ab3dmot_path=None):
        """Run tracking evaluation using submodule eval script."""
        # Use the evaluation script from submodule  
        eval_script = self.submodule_path / "scripts" / "KITTI" / "evaluate.py"
        
        if not eval_script.exists():
            # Fallback to v2x eval script
            eval_script = self.path_manager.base_dir / "third_party" / "ab3dmot" / "v2x" / "eval_tracking.py"
        
        cmd = [
            "python3", str(eval_script),
            "--track_eval_gt_path", track_eval_gt_path,
            "--calib_gt_path", calib_gt_path,
            "--track_results_path", track_results_path,
            "--track_eval_output_path", track_eval_output_path,
            "--cat", cat
        ]
        
        if ab3dmot_path:
            cmd.extend(["--ab3dmot_path", ab3dmot_path])
        else:
            cmd.extend(["--ab3dmot_path", str(self.submodule_path)])
            
        result = subprocess.run(cmd)
        return result.returncode == 0
    
    def run_data_convert(self, input_dir_path, output_dir_path, ori_path):
        """Run data conversion using submodule or plugin implementation."""
        # Use data conversion script from v2x tools
        convert_script = self.path_manager.base_dir / "third_party" / "ab3dmot" / "tools" / "dataset_converter" / "pcdet_result2kitti.py"
        
        if not convert_script.exists():
            print(f"Warning: Data convert script not found at {convert_script}")
            return False
        
        cmd = [
            "python3", str(convert_script),
            "--input-dir-path", input_dir_path,
            "--output-dir-path", output_dir_path,
            "--ori-path", ori_path
        ]
        
        result = subprocess.run(cmd)
        return result.returncode == 0


def main():
    """Command line interface for AB3DMOT adapter."""
    parser = argparse.ArgumentParser(description='AB3DMOT Adapter')
    parser.add_argument('command', choices=['track', 'eval', 'convert'], 
                       help='Command to run')
    parser.add_argument('--dataset', type=str, default='KITTI')
    parser.add_argument('--det_name', type=str, default='openpcdet')
    parser.add_argument('--cat', type=str, default='Car')
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--config-path', type=str, default=None)
    parser.add_argument('--vis-path', type=str, default="")
    
    # Eval specific args
    parser.add_argument('--track_eval_gt_path', type=str, default="")
    parser.add_argument('--calib_gt_path', type=str, default="")
    parser.add_argument('--track_results_path', type=str, default="")
    parser.add_argument('--track_eval_output_path', type=str, default="")
    
    # Convert specific args  
    parser.add_argument('--input-dir-path', type=str, default="", dest='input_dir_path')
    parser.add_argument('--output-dir-path', type=str, default="", dest='output_dir_path')
    parser.add_argument('--ori-path', type=str, default="")
    
    args = parser.parse_args()
    
    adapter = AB3DMOTAdapter()
    
    if args.command == 'track':
        success = adapter.run_main_tracking(
            dataset=args.dataset,
            det_name=args.det_name,
            cat=args.cat,
            input_path=args.input_path,
            output_path=args.output_path,
            config_path=args.config_path,
            vis_path=args.vis_path
        )
    elif args.command == 'eval':
        success = adapter.run_eval_tracking(
            track_eval_gt_path=args.track_eval_gt_path,
            calib_gt_path=args.calib_gt_path, 
            track_results_path=args.track_results_path,
            track_eval_output_path=args.track_eval_output_path,
            cat=args.cat
        )
    elif args.command == 'convert':
        success = adapter.run_data_convert(
            input_dir_path=args.input_dir_path,
            output_dir_path=args.output_dir_path,
            ori_path=args.ori_path
        )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()