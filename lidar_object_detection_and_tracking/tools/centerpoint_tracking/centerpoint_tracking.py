from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys 
import json
import numpy as np
import time
import copy
import argparse
import copy
import json
import os
import numpy as np
# from tools.centerpoint_tracking.tracker import PubTracker as Tracker
from tracker import PubTracker as Tracker
from tqdm import tqdm
import json 
import time
from nuscenes.utils.geometry_utils import transform_matrix
import pickle 
from pyquaternion import Quaternion
import warnings


def parse_args():
    parser = argparse.ArgumentParser(description="Tracking")
    parser.add_argument("--work_dir", help="the dir to save logs and tracking results", default='outputs/tracking/kitti_centerpoint')
    parser.add_argument("--dataset", type=str, default='KITTI')
    parser.add_argument("--det_result", help="the dir to prediction file")
    parser.add_argument("--ego_info", type=str, default='/data/kitti/tracking/training/ego_info')
    parser.add_argument("--max_age", type=int, default=3)
    parser.add_argument("--car", type=float, default=0.7) 
    parser.add_argument("--pedestrian", type=float, default=0.5)  
    parser.add_argument("--cyclist", type=float, default=0.5)  
    parser.add_argument("--score_thresh", type=float, default=0.75)
    parser.add_argument("--obj_type", type=str, default='Car')

    args = parser.parse_args()

    return args

def get_obj(path):
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 

def veh_pos_to_transform(veh_pos):
    "convert CAR pose to two transformation matrix"
    rotation = veh_pos[:3, :3] 
    tran = veh_pos[:3, 3]

    global_from_car = transform_matrix(
        tran, Quaternion(matrix=rotation), inverse=False
    )

    car_from_global = transform_matrix(
        tran, Quaternion(matrix=rotation), inverse=True
    )

    return global_from_car, car_from_global

def reorganize_info(infos):
    new_info = {}
    i = 0
    for info in infos:
        token = str(i * 1e-1)
        new_info[token] = info
        i += 1

    return new_info 

def load_kitti_gt(gt_info_path):
    # load from raw file
    type_str2id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3}
    list_data = []
    # Initialize empty gt_info structure
    gt_info = {
        'bboxes': [],  # List of lists to hold bounding boxes for each frame
        'ids': [],     # List of lists to hold IDs for each frame
        'types': []    # List of lists to hold types for each frame
    }
    
    current_frame_id = -1
    frame_bboxes = []
    frame_ids = []
    frame_types = []

    with open(gt_info_path, 'r') as read_f:
        list_lines = read_f.readlines()
        if len(list_lines) == 0:
            return []
        for line in list_lines:
            line = line.strip('\n').split(' ')
            line[1] = type_str2id[line[1]]
            line = [float(i) for i in line]
            kitti_data = np.array(line)
            # x, y, z, o, l, w, h
            frame_id = int(kitti_data[0])
            bbox = [kitti_data[17], kitti_data[18], kitti_data[19], kitti_data[20], kitti_data[12], kitti_data[11], kitti_data[10]]
            ids = kitti_data[1]
            types = kitti_data[2]
            if frame_id != current_frame_id:
                if current_frame_id != -1:
                    gt_info['bboxes'].append(bbox)
                    gt_info['ids'].append(ids)
                    gt_info['types'].append(types)
                current_frame_id = frame_id
                frame_bboxes = []
                frame_ids = []
                frame_types = []
            
            frame_bboxes.append(bbox)
            frame_ids.append(ids)
            frame_types.append(types)
        # Append the last frame's data
        if current_frame_id != -1:
            gt_info['bboxes'].append(frame_bboxes)
            gt_info['ids'].append(frame_ids)
            gt_info['types'].append(frame_types)
    return gt_info



def transform_box(box, pose):
    """Transforms 3d upright boxes from one frame to another.
    Args:
    box: [..., N, 7] boxes.
    from_frame_pose: [...,4, 4] origin frame poses.
    to_frame_pose: [...,4, 4] target frame poses.
    Returns:
    Transformed boxes of shape [..., N, 7] with the same type as box.
    """
    transform = pose 
    heading = box[..., -1] + np.arctan2(transform[..., 1, 0], transform[..., 0,
                                                                    0])
    center = np.einsum('...ij,...nj->...ni', transform[..., 0:3, 0:3],
                    box[..., 0:3]) + np.expand_dims(
                        transform[..., 0:3, 3], axis=-2)

    velocity = box[..., [6, 7]] 

    velocity = np.concatenate([velocity, np.zeros((velocity.shape[0], 1))], axis=-1) # add z velocity

    velocity = np.einsum('...ij,...nj->...ni', transform[..., 0:3, 0:3],
                    velocity)[..., [0, 1]] # remove z axis 

    return np.concatenate([center, box[..., 3:6], velocity, heading[..., np.newaxis]], axis=-1)

def label_to_name(label):
    if label == 1:
        return "Car"
    elif label == 2 :
        return "Pedestrian"
    elif label == 3:
        return "Cyclist"
    else:
        raise NotImplemented()

def sort_detections(detections):
    indices = [] 

    for det in detections:
        seq_id = int(det['seq_name'])
        frame_id= int(det['frame_id'])

        idx = seq_id * 1000 + frame_id
        indices.append(idx)

    rank = list(np.argsort(np.array(indices)))

    detections = [detections[r] for r in rank]

    return detections

def convert_detection_to_global_box(detections, ego_infos):
    ret_list = [] 

    detection_results = {} # copy.deepcopy(detections)

    for seq_name in tqdm(ego_infos.keys()):
        for index in range(len(detections[seq_name]['frame_id'])):
            detection = detections[seq_name]
            detection_results[seq_name] = copy.deepcopy(detection)

            ego_info = ego_infos[seq_name][index]
            print('ego_info shape is : ', ego_info.shape)
            # pose = np.ndarray(ego_info).reshape(-1, 4, 4)
            pose = ego_info
            box3d = detection["box3d_lidar"][index]
            labels = detection["label_preds"][index]
            scores = detection['scores'][index]

            box3d = transform_box(box3d, pose)

            frame_id = detection['frame_id'][index]

            num_box = len(box3d)

            anno_list =[]
            for i in range(num_box):
                anno = {
                    'translation': box3d[i, :3],
                    'velocity': box3d[i, [6, 7]],
                    'detection_name': label_to_name(labels[i]),
                    'score': scores[i], 
                    'box_id': i 
                }

                anno_list.append(anno)

            ret_list.append({
                'seq_name': seq_name, 
                'frame_id':int(frame_id),
                'global_boxs': anno_list,
                'timestamp': frame_id*1e-1
            })

    sorted_ret_list = sort_detections(ret_list)

    return sorted_ret_list, detection_results 

def load_kitti_ego_pose(file):
    # load from raw file
    ego_infos_npz = np.load(file, allow_pickle=True)
    ego_infos = ego_infos_npz['pose_data']
    return ego_infos

def load_kitti_detection(file):
    # load from raw file
    type_str2id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3}
    # Initialize empty gt_info structure
    detection = {
        'frame_id': [],
        'box3d_lidar': [],  
        'label_preds': [],     
        'scores': []    
    }
    
    current_frame_id = -1
    frame_bboxes = []
    frame_types = []
    frame_scores = []

    with open(file, 'r') as read_f:
        list_lines = read_f.readlines()
        if len(list_lines) == 0:
            return []
        last_x, last_y, vel_x, vel_y = 0.0, 0.0, 0.0, 0.0
        for line in list_lines:
            line = line.strip('\n').split(' ')
            line[1] = type_str2id[line[1]]
            line = [float(i) for i in line]
            kitti_data = np.array(line)
            # x, y, z, o, l, w, h
            frame_id = int(kitti_data[0])
            bbox_tmp = [kitti_data[17], kitti_data[18], kitti_data[19], kitti_data[20], kitti_data[12], kitti_data[11], kitti_data[10]]
            vel_x = (bbox_tmp[0] - last_x) / 1e-1
            vel_y = (bbox_tmp[1] - last_y) / 1e-1
            bbox = [bbox_tmp[0], bbox_tmp[1], bbox_tmp[2], bbox_tmp[6], bbox_tmp[5], bbox_tmp[4], vel_x, vel_y, bbox_tmp[3]]
            score = kitti_data[22]
            type = kitti_data[1]
            if frame_id != current_frame_id:
                if current_frame_id != -1:
                    detection["box3d_lidar"].append(frame_bboxes)
                    detection["label_preds"].append(frame_types)
                    detection['scores'].append(frame_scores)
                    detection['frame_id'].append(current_frame_id)
                current_frame_id = frame_id
                frame_bboxes = []
                frame_types = []
                frame_scores = []
            
            frame_bboxes.append(bbox)
            frame_types.append(type)
            frame_scores.append(score)
            last_x = bbox[0]
            last_y = bbox[1]
        # Append the last frame's data
        if current_frame_id != -1:
            detection["box3d_lidar"].append(frame_bboxes)
            detection["label_preds"].append(frame_types)
            detection['scores'].append(frame_scores)
            detection['frame_id'].append(current_frame_id)
    return detection

def save_results(tracking_id, box3d_lidar, label_pred, score, eval_file, frame, score_threshold):
    type_str2id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3}
    # box3d in the format of h, w, l, x, y, z, theta in camera coordinate
    id_tmp = tracking_id
    type_tmp = type_str2id[label_pred]
    conf_tmp = score
    lidar_xyzr = box3d_lidar[0:4]
    l = box3d_lidar[4]
    w = box3d_lidar[5]
    h = box3d_lidar[6]

    # save in tracking format, for 3D MOT evaluation
    if conf_tmp >= score_threshold:
        str_to_srite = '%06d %s %d 0 0 -1 -1 -1 -1 -1 %f %f %f -1 -1 -1 -1 %f %f %f %f -1 -1 %f -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1\n' % (
            int(frame), type_tmp, id_tmp, h, w, l, lidar_xyzr[0],lidar_xyzr[1],lidar_xyzr[2],lidar_xyzr[3], conf_tmp)
        eval_file.write(str_to_srite)

def main():
    args = parse_args()
    print('Deploy OK')

    max_dist = {
        'Car': args.car,
        'Pedestrian': args.pedestrian,
        'Cyclist': args.cyclist
    }

    tracker = Tracker(max_age=args.max_age, max_dist=max_dist, score_thresh=args.score_thresh)

    seq_eval =sorted(os.listdir(args.det_result))
    predictions = {}
    ego_infos = {}
    for seq_file in seq_eval:
        seq_name = seq_file.split('/')[-1].split('.')[0]
        print('cur seg name is : ',seq_name)
        seq_file = os.path.join(args.det_result, f'{seq_name}.txt')  # ./data/KITTI/detection/pointrcnn_Car_val/0000.txt
        prediction = load_kitti_detection(seq_file)
        pose_file = os.path.join(args.ego_info, f'{seq_name}.npz')
        ego_info = load_kitti_ego_pose(pose_file)

        predictions[seq_name] = prediction
        ego_infos[seq_name] = ego_info

    global_preds, detection_results = convert_detection_to_global_box(predictions, ego_infos)
    size = len(global_preds)

    print("Begin Tracking {} frames\n".format(size))

    predictions = {} 

    for i in tqdm(range(size)):
        pred = global_preds[i]
        seq_name = pred['seq_name']

        # reset tracking after one video sequence
        if pred['frame_id'] == 0:
            tracker.reset()
            last_time_stamp = pred['timestamp']

        time_lag = (pred['timestamp'] - last_time_stamp) 
        last_time_stamp = pred['timestamp']

        current_det = pred['global_boxs']

        outputs = tracker.step_centertrack(current_det, time_lag)
        tracking_ids = []
        box_ids = [] 

        for item in outputs:
            if item['active'] == 0:
                continue 
            
            box_ids.append(item['box_id'])
            tracking_ids.append(item['tracking_id'])

        # now reorder 
        detection = detection_results[seq_name]

        remained_box_ids = np.array(box_ids)

        track_result = {} 

        # store box id 
        track_result['tracking_ids']= np.array(tracking_ids)   

        # store box parameter 
        track_result['box3d_lidar'] = detection['box3d_lidar'][remained_box_ids]

        # store box label 
        track_result['label_preds'] = detection['label_preds'][remained_box_ids]

        # store box score 
        track_result['scores'] = detection['scores'][remained_box_ids]

        track_result['frame_id'] = detection['frame_id']

        predictions['seq_name'] = track_result 

    os.makedirs(args.work_dir, exist_ok=True)
    # save prediction files to args.work_dir 
    for seq_prediction in predictions:
        seq_name = seq_prediction['seq_name']
        for tracking_ids, box3d_lidar, label_preds, scores, frame_id in zip(seq_prediction['tracking_ids'], seq_prediction['box3d_lidar'], seq_prediction['label_preds'], seq_prediction['scores'], seq_prediction['frame_id']):
            save_trk_file = open(os.path.join(args.work_dir, prediction['seq_name'] + '.txt'), 'w')
            eval_file = open(os.path.join(args.work_dir, prediction['seq_name'] + '.txt'), 'w')
            frame = frame_id
            score_threshold = args.score_threshold
            for idx in range(len(tracking_ids)):
                save_results(tracking_ids[idx], box3d_lidar[idx], label_preds[idx], scores[idx], save_trk_file, eval_file, frame, score_threshold)
                save_trk_file.close()

if __name__ == '__main__':
    main()
