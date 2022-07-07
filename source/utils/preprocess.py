import os
os.environ['DISPLAY'] = ':1'

from tqdm import tqdm

import numpy as np
import open3d as o3d

import utils.pointcloud as pointcloud
from utils.utility import set_dict
from tqdm import tqdm

            
def gt_generate(data_dict, config):
    """
    Generate ground truth overlap with gt pointcloud

    Args:
        data_dict
        config
    """
    
    if config['datatype'] == 'ScanNet':
        save_path = data_dict['gt_path']
        if not os.path.exists(save_path) or config['debug_mode']:
        
            pcd_gt_list = []
            intrinsic = config['intrinsic']
            depth_scale=config['depth_scale']
            depth_trunc=config['depth_trunc']
            voxel_size=config['gt_voxel_size']
            
            for i in tqdm(range(config['n_frames'])):
                gt_pcd_v_path = data_dict['gt_pcd_v'][i]
                if os.path.exists(gt_pcd_v_path):
                    with open(gt_pcd_v_path, 'rb') as f:
                        pcd_array = np.load(f)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pcd_array[:,:3])
                    pcd.colors = o3d.utility.Vector3dVector(pcd_array[:,3:])
                    pcd_gt_list.append(pcd)
        
                else :
                    depth_path = data_dict['depth'][i]     
                    color_path = data_dict['color'][i]
                    extrinsic = data_dict['gtpose'][i]
                    depth_raw = o3d.io.read_image(depth_path)
                    color_raw = o3d.io.read_image(color_path)
                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                                                                color_raw,
                                                                depth_raw,
                                                                depth_scale=depth_scale,
                                                                convert_rgb_to_intensity=False,
                                                                depth_trunc=depth_trunc)
                    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic=intrinsic).voxel_down_sample(config['gt_voxel_size'])
                    pcd = pcd.transform(extrinsic)
                    pcd_gt_list.append(pcd)
                    
                    xyz = np.array(pcd.points)
                    feats = np.array(pcd.colors)
                    pcd_np = np.hstack([xyz, feats])
                    with open(gt_pcd_v_path, 'wb')as f:
                        np.save(f, pcd_np)
        
            if config['visualize']:
                o3d.visualization.draw_geometries(pcd_gt_list)
                
            gt_mat = np.eye(len(pcd_gt_list))
            pair_list = []
            for s_idx in range(len(pcd_gt_list)-1):
                for t_idx in range(s_idx+1, len(pcd_gt_list)):
                    pair_list.append((s_idx, t_idx))
            
            for (s_idx, t_idx) in tqdm(pair_list):
                s_pcd = pcd_gt_list[s_idx]
                t_pcd = pcd_gt_list[t_idx]
                if len(s_pcd.points)<10 or len(t_pcd.points)<10:
                    gt = 0
                else:
                    gt = pointcloud.compute_overlap_ratio_mean(s_pcd, t_pcd, np.eye(4), voxel_size)
                gt_mat[s_idx, t_idx] = gt
                gt_mat[t_idx, s_idx] = gt

            with open(save_path, 'wb')as f:
                np.save(f, gt_mat)
    
    else:
        assert NotImplementedError
        
        
def preprocess(scene_path, args):
    data_dict, config = set_dict(scene_path, args)
    gt_generate(data_dict, config)