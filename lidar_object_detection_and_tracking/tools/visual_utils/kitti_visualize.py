import numpy as np
import open3d as o3d
import struct

def load_kitti_bin(bin_path):
    """
    Load a KITTI point cloud from a .bin file.
    :param bin_path: Path to the .bin file.
    :return: Numpy array of shape (N, 4) where N is the number of points.
             Each point has [x, y, z, intensity].
    """
    size_float = 4
    list_pcd = []
    with open(bin_path, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    return np.asarray(list_pcd)

def parse_kitti_label(label_path):
    """
    Parse a KITTI label file to extract 3D bounding box information.
    :param label_path: Path to the .txt label file.
    :return: List of dictionaries, each containing the parameters of a 3D bounding box.
    """
    bboxes = []
    with open(label_path, 'r') as f:
        for line in f:
            fields = line.strip().split(' ')
            if len(fields) < 15:
                continue
            # Extract relevant fields
            cls_name = fields[0]
            dimensions = [float(fields[8]), float(fields[9]), float(fields[10])]  # height, width, length
            location = [float(fields[11]), float(fields[12]), float(fields[13])]  # x, y, z (center of the box)
            rotation_y = float(fields[14])  # Rotation around the y-axis (up-axis)
            
            bbox_info = {
                'class': cls_name,
                'dimensions': dimensions,  # [height, width, length]
                'location': location,      # [x, y, z]
                'rotation_y': rotation_y   # Rotation around y-axis
            }
            bboxes.append(bbox_info)
    return bboxes

def load_calib(calib_path):
    """
    Load the KITTI calibration file and extract transformation matrices.
    :param calib_path: Path to the .txt calibration file.
    :return: Dictionary containing the calibration matrices.
    """
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            key, value = line.split(':', 1)
            try:
                calib[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    
    # Reshape the matrices
    P0 = calib['P0'].reshape(3, 4)
    P1 = calib['P1'].reshape(3, 4)
    P2 = calib['P2'].reshape(3, 4)
    P3 = calib['P3'].reshape(3, 4)
    R0_rect = calib['R0_rect'].reshape(3, 3)
    Tr_velo_to_cam = calib['Tr_velo_to_cam'].reshape(3, 4)

    return {
        'P0': P0,
        'P1': P1,
        'P2': P2,
        'P3': P3,
        'R0_rect': R0_rect,
        'Tr_velo_to_cam': Tr_velo_to_cam
    }

def camera_to_lidar(box_center, box_rotation, calib):
    """
    Transform a 3D bounding box from camera coordinates to LiDAR coordinates.
    :param box_center: Center of the bounding box in camera coordinates [x, y, z].
    :param box_rotation: Rotation angle around the y-axis in camera coordinates.
    :param calib: Calibration dictionary containing transformation matrices.
    :return: Transformed center and rotation in LiDAR coordinates.
    """
    # Get transformation matrices
    R0_rect = np.eye(4)
    R0_rect[:3, :3] = calib['R0_rect']
    Tr_velo_to_cam = np.vstack((calib['Tr_velo_to_cam'], [0, 0, 0, 1]))
    Tr_cam_to_velo = np.linalg.inv(Tr_velo_to_cam)

    # Convert center from camera coordinates to LiDAR coordinates
    box_center_camera = np.array(box_center + [1])  # Homogeneous coordinates
    box_center_lidar = Tr_cam_to_velo @ R0_rect @ box_center_camera

    # Adjust rotation: rotation_y in camera coordinates -> rotation_z in LiDAR coordinates
    box_rotation_lidar = -box_rotation - np.pi / 2

    return box_center_lidar[:3], box_rotation_lidar

def create_open3d_box(bbox_info, calib, color=(1, 0, 0)):
    """
    Create an Open3D OrientedBoundingBox from KITTI 3D bounding box info.
    :param bbox_info: Dictionary containing bounding box parameters.
    :param calib: Calibration dictionary.
    :param color: Color of the bounding box (default: red).
    :return: Open3D OrientedBoundingBox object.
    """
    # Transform center and rotation from camera coordinates to LiDAR coordinates
    center_camera = bbox_info['location']
    rotation_y = bbox_info['rotation_y']
    center_lidar, rotation_z = camera_to_lidar(center_camera, rotation_y, calib)

    dimensions = bbox_info['dimensions']  # [height, width, length]
    extent = [dimensions[2], dimensions[1], dimensions[0]]  # [length, width, height]

    # Create rotation matrix for Open3D
    R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, rotation_z])

    # Create the bounding box
    box = o3d.geometry.OrientedBoundingBox(center=center_lidar, R=R, extent=extent)
    box.color = color
    return box

def visualize_kitti_point_cloud_and_boxes(pcd_path, label_path, calib_path):
    """
    Visualize a KITTI point cloud and its 3D bounding boxes.
    :param pcd_path: Path to the .bin point cloud file.
    :param label_path: Path to the .txt label file.
    :param calib_path: Path to the .txt calibration file.
    """
    # Load point cloud
    pcd = load_kitti_bin(pcd_path)
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)

    # Load calibration
    calib = load_calib(calib_path)

    # Parse labels and create bounding boxes
    bboxes_info = parse_kitti_label(label_path)
    boxes = [create_open3d_box(box_info, calib) for box_info in bboxes_info]

    # Combine all geometries
    geometries = [pcd_o3d] + boxes

    # Visualize
    o3d.visualization.draw_geometries(geometries)

if __name__ == "__main__":
    # Specify paths to your KITTI files
    pcd_file = 'path_to_your_point_cloud.bin'  # Replace with your .bin file path
    label_file = 'path_to_your_labels.txt'     # Replace with your .txt label file path
    calib_file = 'path_to_your_calib.txt'      # Replace with your .txt calibration file path

    # Call the visualization function
    visualize_kitti_point_cloud_and_boxes(pcd_file, label_file, calib_file)