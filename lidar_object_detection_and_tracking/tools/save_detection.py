import argparse
import glob
from pathlib import Path
import os
import shutil
try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--output_file_path', type=str, default='./output/demo', help='specify the path of your output data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def save_pred_label(pred_dicts, output_file_path_label):
    lines = []
    len_predt = len(pred_dicts[0]['pred_boxes'])
    pred_obj_list = pred_dicts[0]['pred_boxes'].cpu().data.numpy()
    pred_cls_list = pred_dicts[0]['pred_labels'].cpu().data.numpy()
    pred_score_list = pred_dicts[0]['pred_scores'].cpu().data.numpy()
    class_to_name = {
        1: 'Car',
        2: 'Pedestrian',
        3: 'Cyclist',
        4: 'Van',
        5: 'Person_sitting',
        6: 'Truck'
    }
    
    for i in range(len_predt):
        bounding_box = [0, 0, 0, 0]
        # Float from 0 (non-truncated) to 1 (truncated), where truncated refers to the object leaving image boundaries
        # NOTE: truncation set to 0.0 as we do not have any information about it
        truncated = 0.00
        # 0 = fully visible, 1 = partly occluded, 2 = largely occluded, 3 = unknown
        # NOTE: occlusion set to 3 as we do not have any information about it
        occluded = 3
        # Observation angle of object, ranging [-pi..pi]
        # NOTE: observation angle (alpha) set to 0.0 as we do not have any information about it
        alpha = 0.00
        pred_obj = pred_obj_list[i,:]
        pred_cls = class_to_name[pred_cls_list[i]]
        pred_score = pred_score_list[i]
        if pred_score < 0.3:
            continue
        # 345 lhw
        height = pred_obj[4]
        width = pred_obj[5]
        length = pred_obj[3]
        x_center = pred_obj[0]
        y_center = pred_obj[1]
        z_center = pred_obj[2]
        yaw = pred_obj[6]
        line = f"{pred_cls} {round(truncated, 2)} {occluded} {round(alpha, 2)} " + \
                f"{round(bounding_box[0], 2)} {round(bounding_box[1], 2)} {round(bounding_box[2], 2)} " + \
                f"{round(bounding_box[3], 2)} {round(height, 2)} {round(width, 2)} {round(length, 2)} " + \
                f"{round(x_center, 2)} {round(y_center, 2)} {round(z_center, 2)} {round(yaw, 2)}\n"
        lines.append(line)
    fp_label = open(output_file_path_label, 'a')
    fp_label.writelines(lines)
    fp_label.close()
    
def save_point_cloud(data_dict, output_file_path_point_cloud):
    point_cloud = data_dict['points'][:, 1:].cpu().data.numpy()
    bin_format = point_cloud
    bin_format.tofile(os.path.join(output_file_path_point_cloud))   
    
def main():
    args, cfg = parse_config()
    output_path = args.output_file_path
    output_label_path = os.path.join(output_path, 'label')
    os.makedirs(output_label_path, exist_ok=True)
    output_pointcloud_path = os.path.join(output_path, 'pointcloud')
    os.makedirs(output_pointcloud_path, exist_ok=True)
    output_calib_path = os.path.join(output_path, 'calib')
    os.makedirs(output_calib_path, exist_ok=True)
    file_idx, _ = os.path.splitext(os.path.basename(args.data_path))
    calib_path = Path('/mnt/data/kitti/r_livit_detection/training/calib')

    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            save_pred_label(pred_dicts, os.path.join(output_label_path, f'{file_idx}.txt'))
            save_point_cloud(data_dict, os.path.join(output_pointcloud_path, f'{file_idx}.bin'))
            shutil.copy(os.path.join(calib_path, f'{file_idx}.txt'),output_calib_path)

    logger.info('Save detection result done.')


if __name__ == '__main__':
    main()
