import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import json
from pathlib import Path

class InferenceDataset(DatasetTemplate):
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
        data_file_list = glob.glob(str(root_path / f'**/*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

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
        sequence_id = float(os.path.basename(os.path.dirname(self.sample_file_list[index])))
        timestamp = float(os.path.splitext(os.path.basename(self.sample_file_list[index]))[0])
        input_dict = {
            'points': points,
            'frame_id': index,
            'sequence_id': sequence_id,
            'timestamp': timestamp
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for inference')
    parser.add_argument('--data_path', type=str, default='inference_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--save_path', type=str, default='/home/user/workspace/output/tools/cfgs/r_livit_models/tracking',
                        help='specify the output directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def torch_tensor_to_numpy(torch_tensor):
    """
    Convert a torch tensor to numpy.

    Parameters
    ----------
    torch_tensor : torch.Tensor

    Returns
    -------
    A numpy array.
    """
    return torch_tensor.numpy() if not torch_tensor.is_cuda else \
        torch_tensor.cpu().detach().numpy()


def save_prediction_json(pred_tensor, pred_score_tensor, pred_label_tensor, sequence_id, timestamp, save_path):
    """
    Save prediction to json file.
    """
    try:
        bbox = torch_tensor_to_numpy(pred_tensor).tolist()
        score = torch_tensor_to_numpy(pred_score_tensor).tolist()
        label = torch_tensor_to_numpy(pred_label_tensor).tolist()

        pred = {
            'bbox': bbox,
            'score': score,
            'label': label
        }
    except:
        pred = {
            'bbox': [],
            'score': [],
            'label': []
        }
    print('saving')
    sequence_id = int(sequence_id[0,0])
    save_path = os.path.join(save_path, f'{sequence_id:04d}')
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, '%06d_pred.json' % (timestamp)), 'w') as f:
        json.dump(pred, f, indent=4)

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()

    save_path_dir = Path(args.save_path)
    save_path_dir.mkdir(parents=True, exist_ok=True)

    inference_dataset = InferenceDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(inference_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=inference_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(inference_dataset):
            logger.info(f'Inferenced sample index: \t{idx + 1}')
            data_dict = inference_dataset.collate_batch([data_dict])
            sequence_id = inference_dataset.collate_batch([data_dict])['sequence_id']
            timestamp = inference_dataset.collate_batch([data_dict])['timestamp']
            output_dir = save_path_dir
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            save_prediction_json(pred_dicts[0]['pred_boxes'], pred_dicts[0]['pred_scores'], pred_dicts[0]['pred_labels'], sequence_id, timestamp, output_dir)
            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )

    logger.info('Inference done.')


if __name__ == '__main__':
    main()