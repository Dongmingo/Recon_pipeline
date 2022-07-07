from ast import Not
import os
import copy

import numpy as np
import open3d as o3d

from PIL import Image
from scipy.spatial.transform import Rotation as R

def load_image_size(datatype, input_dir):
    if datatype == 'ScanNet':
        depth_path = os.path.join(input_dir, 'depth', '0.png')
        (w, h) = Image.open(depth_path).size
    else: assert NotImplementedError
    
    return (w, h)


def load_intrinsic(datatype, scene_path, img_w, img_h):
    if datatype == 'ScanNet':        # ScanNet, 4 x 4 intrinsic matrix
        intrinsic_path = os.path.join(scene_path, 'intrinsic/intrinsic_depth.txt')
        with open(intrinsic_path, 'r') as f:
            data = f.read()
            lst = data.replace('\n', ' ').split(' ')
            mat_intrinsic = np.reshape([float(i) for i in lst[:16]], (4, 4))
        fx, fy, cx, cy = mat_intrinsic[0, 0], mat_intrinsic[1, 1], mat_intrinsic[0, 2], mat_intrinsic[1, 2]
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()  # o3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault
        pinhole_camera_intrinsic.set_intrinsics(width=img_w, height=img_h, fx=fx, fy=fy, cx=cx, cy=cy)
        
    # elif datatype == '3DMatch':        # 3DMatch, 3 x 3 intrinsic matrix
    #     intrinsic_path = os.path.join(scene_path, '../frame-intrinsic')
    #     with open(intrinsic_path, 'r') as f:
    #         data = f.read()
    #         lst = data.replace('\n', '').split('\t')
    #         mat_intrinsic = np.reshape([float(i) for i in lst[:9]], (3, 3))
    #     fx, fy, cx, cy = mat_intrinsic[0, 0], mat_intrinsic[1, 1], mat_intrinsic[0, 2], mat_intrinsic[1, 2]
    #     pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()  # o3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault
    #     pinhole_camera_intrinsic.set_intrinsics(width=img_w, height=img_h, fx=fx, fy=fy, cx=cx, cy=cy)
    
    # elif datatype == 'ETH3D':        # ETH3D, fx, fy, cx, cy
    #     intrinsic_path = os.path.join(scene_path, intrinsic)
    #     f = open(intrinsic_path)
    #     line = f.readline()
    #     [fx, fy, cx, cy] = [float(x) for x in line.strip('\n').split(' ')]
            
    #     pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()  
    #     pinhole_camera_intrinsic.set_intrinsics(width=img_w, height=img_h, fx=fx, fy=fy, cx=cx, cy=cy)
        
    # elif datatype == 'ICL-NUIM':            # ICL-NUIM, 640 x 480, 481.20, -480.00, 319.5, 239.5
    #     pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()  
    #     pinhole_camera_intrinsic.set_intrinsics(width=img_w, height=img_h, fx=481.20, fy=-480.00, cx=319.50, cy=239.50)
    
    else :
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)  # Not included intrinsic, 640 x 480, 525, 525, 319.5, 239.5
        assert NotImplementedError, f"{datatype} intrinsic load not Implemented"
        
    print('camera intrinsic matrix : \n',pinhole_camera_intrinsic.intrinsic_matrix)
    return pinhole_camera_intrinsic

def load_extrinsic(gt_path, datatype, index):
    '''
    From each dataset's different annotation, load all frames absolute extrinsic matrix(4x4) with proper interval(frame_jump) if it exists
    Input : str_list to return index, gt_path, frame_jump for frame interval
    Output : matching_indices (N verctor), extrinsic (Nx4x4 extrinsic)
    '''
    if datatype == 'ScanNet':       # ScanNet
        pose_path = os.path.join(gt_path, str(index)+'.txt')
        with open(pose_path, 'r')as f:
            data = f.read()
            lst = data.replace('\n', ' ').split(' ')
            mat_extrinsic = np.reshape([float(i) for i in lst[:16]], (4, 4))
              
    # elif os.path.splitext(gt_path)[1] == '.txt':        # ETH3D(meta), TUM(meta3 lines), ICL_NUIM ts, tx, ty, tz, qx, qy, qz, qw
    #     total_extrinsic = []
    #     total_indices = []
    #     matching_indices = []
    #     extrinsic = []
    #     lines = open(os.path.join(dir_path, gt_path), 'r').readlines()
    #     cnt = 0
    #     for line in lines[:5]:
    #         if '#' in line:
    #             cnt += 1
    #     for line in lines[cnt:]:
    #         [index, tx, ty, tz, qx, qy, qz, qw] = [float(x) for x in line.strip('\n').split(' ')]
    #         total_indices.append(index)
    #         rot_np = np.array(R.from_quat([qx, qy, qz, qw]).as_matrix())
    #         ext_mat = np.hstack([np.vstack([rot_np,[0,0,0]]),[[tx],[ty],[tz],[1]]])
    #         total_extrinsic.append(ext_mat)
    #     total_extrinsic = np.array(total_extrinsic)
    #     total_indices = np.array(total_indices)
    #     ts_indices = np.array([float(x) for x in str_list])
    #     ts_match, total_match = matching_time_indices(ts_indices, total_indices)
    #     for i in range(0, len(ts_match), frame_jump):
    #         matching_indices.append(ts_match[i])
    #         extrinsic.append(total_extrinsic[total_match[i]])
    #     extrinsic = np.array(extrinsic)
            
    # elif os.path.splitext(gt_path)[1] == '.log' :      # Redwood (scene_name).txt, meta data line + following 4x4 extrinsic -> 5x4xN
    #     matching_indices = []
    #     extrinsic = []
    #     lines = open(os.path.join(dir_path, gt_path), 'r').readlines()
    #     num_f = len(lines)//5
    #     for i in range(0, num_f, frame_jump):
    #         mat_ext = np.eye(4)
    #         line_meta = lines[5*i]
    #         [index, _, _] = line_meta.strip('\n').split(' ')
    #         for j in range(1,5):
    #             line = lines[5*i+j]
    #             mat_ext[j-1, :] = np.fromstring(line, dtype=float, sep=' \t')
    #         if index in str_list:
    #             matching_indices.append(str_list.index(index))
    #             extrinsic.append(mat_ext)
    #     extrinsic = np.array(extrinsic)
        
    else:
        assert NotImplementedError, f"{datatype}"
    return mat_extrinsic    # 4x4 extrinsic

def matching_time_indices(stamps_1: np.ndarray, stamps_2: np.ndarray,
                        max_diff: float = 0.01,
                        offset_2: float = 0.0):
    matching_indices_1 = []
    matching_indices_2 = []
    stamps_2 = copy.deepcopy(stamps_2)
    stamps_2 += offset_2
    for index_1, stamp_1 in enumerate(stamps_1):
        diffs = np.abs(stamps_2 - stamp_1)
        index_2 = int(np.argmin(diffs))
        if diffs[index_2] <= max_diff:
            matching_indices_1.append(index_1)
            matching_indices_2.append(index_2)
    return matching_indices_1, matching_indices_2