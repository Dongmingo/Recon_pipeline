"""
Utilities
"""
import os
import glob
import logging

import numpy as np
import open3d as o3d

from PIL import Image

from utils.load_data import load_intrinsic, load_image_size, load_extrinsic


def set_dict(scene, args):
    input_dir = scene
    logging.info(f"Setting Configureation for reconstruct scene : {input_dir}")    
    config = {}
    config['python_multi_threading'] = args.multi_thread
    config['debug_mode'] = args.debug
    config['visualize'] = args.visualize
    config['datatype'] = args.datatype
    
    (w, h) = load_image_size(args.datatype, input_dir)
    intrinsic = load_intrinsic(args.datatype, input_dir, w, h)
    config['intrinsic'] = intrinsic
    config['img_size'] = (w,h)
    config['prone_th'] = args.prone_th
    config['depth_scale'] = set_depth_scale(args.datatype)
    config['depth_trunc'] = args.depth_trunc
    config['max_correspondence_coarse'] = args.voxel_size * 15.0
    config['max_correspondence_fine'] = args.voxel_size * 1.0
    config['voxel_size'] = args.voxel_size
    config['gt_voxel_size'] = args.gt_voxel_size
    config['pyramid_level'] = o3d.utility.IntVector([args.py_c, args.py_m, args.py_f])
    config['tsdf_cubic_size'] = args.tsdf_cubic_size
    config['icp_method'] = args.icp_method
    config['global_registration'] = args.global_registration
    
    config['frame_jump'] = args.frame_jump
    config['keyframe_rate'] = args.keyframe_rate
    total_f = total_frame(args.datatype, input_dir)
    config["start_index"] = args.start_index
    config["end_index"] = args.end_index if args.end_index != 0 else total_f-1
    num_f = (config["end_index"]-config["start_index"]) // args.frame_jump + 1
    config['n_frames'] = num_f
    logging.info(f"Number of frames to use : {num_f}")

    config['reg_option'] = o3d.pipelines.odometry.OdometryOption(
                                            max_depth_diff = config['max_correspondence_coarse'],
                                            iteration_number_per_pyramid_level = config['pyramid_level'])
    
    config['opt_option'] = o3d.pipelines.registration.GlobalOptimizationOption(
                                            max_correspondence_distance=config['max_correspondence_fine'],
                                            edge_prune_threshold=config['prone_th'],                
                                            reference_node=0)
    
    config['overlap_th'] = args.overlap_th
    config['overlap_buffer'] = args.overlap_buffer
    
    logging.info("Setting data path")
    
    data_dict = {
        'index' : [],
        'color' : [],
        'depth' : [],
        'gtpose' : [],
        'gt_pcd_v' : [],
    }
    
    data_dict['scene_path'] = input_dir
    
    gen_folder = os.path.join(input_dir, 'generated_data')
    os.makedirs(gen_folder, exist_ok=True)
    data_dict['generated_path'] = gen_folder
    
    if args.datatype == 'ScanNet':
        os.makedirs(os.path.join(gen_folder, 'gt_voxelized_pcd'), exist_ok=True)
        color_path = os.path.join(gen_folder,'rgb_resize')
        depth_path = os.path.join(input_dir, 'depth')
        gt_path = os.path.join(input_dir, 'pose')
        
        for index in range(config['start_index'], config['end_index']+1, config['frame_jump']):
            data_dict['index'].append(index)
            data_dict['color'].append(os.path.join(color_path, str(index)+'.png'))
            data_dict['depth'].append(os.path.join(depth_path, str(index)+'.png'))
            data_dict['gt_pcd_v'].append(os.path.join(gen_folder,'gt_voxelized_pcd', str(index)+'.bin'))
            data_dict['gtpose'].append(load_extrinsic(gt_path, config['datatype'], index))
    
    if args.output:
        data_dict['output'] = args.output_dir
    else:
        output_folder = os.path.join(gen_folder, 'results')
        os.makedirs(output_folder, exist_ok=True)
        
        data_dict['output'] = output_folder
    
    data_dict['embed_path'] = os.path.join(gen_folder, 'embeddings.bin')
    with open(data_dict['embed_path'], 'rb')as f:
        embed_t = np.load(f)
    data_dict['embeddings'] = embed_t[data_dict['index']]
    
    jumpname = str(config['frame_jump'])+'_'+str(config['keyframe_rate']*config['frame_jump'])
    
    obj_folder = os.path.join(gen_folder, 'obj')
    os.makedirs(obj_folder, exist_ok=True)
    data_dict['legacy_obj'] = os.path.join(obj_folder, 'legacy_'+jumpname+'.obj')
                          
    ply_folder =os.path.join(gen_folder, 'ply')         
    os.makedirs(ply_folder, exist_ok=True)
    data_dict['ply_template'] = os.path.join(ply_folder, jumpname+'.ply') 
    
    edge_folder = os.path.join(gen_folder, 'edge')
    os.makedirs(os.path.join(edge_folder), exist_ok=True)
    data_dict['legacy_succ_reg'] = os.path.join(edge_folder, 'legacy_success_'+jumpname+'.bin')
    data_dict['legacy_final_edge'] = os.path.join(edge_folder, 'legacy_final_'+jumpname+'.bin')
    data_dict['edge_candidate'] = os.path.join(edge_folder, 'edge_candidate_'+jumpname+'.pickle')
    
    reg_eval_folder = os.path.join(gen_folder, 'reg_eval')
    os.makedirs(reg_eval_folder, exist_ok=True)
    
    os.makedirs(os.path.join(gen_folder, 'posegraph'), exist_ok=True)
    data_dict['legacy_posegraph_b'] = os.path.join(gen_folder, 'posegraph', 'legacy_b_'+jumpname+'.json')  
    data_dict['legacy_posegraph_a'] = os.path.join(gen_folder, 'posegraph', 'legacy_a_'+jumpname+'.json')  
    data_dict['embed_posegraph_b'] = os.path.join(gen_folder, 'posegraph', 'embed_b_'+jumpname+'.json')  
    data_dict['embed_posegraph_a'] = os.path.join(gen_folder, 'posegraph', 'embed_a_'+jumpname+'.json')  
    
    se_jumpname = f"{config['start_index']}-{config['end_index']}_{config['frame_jump']}"
    overlap_folder = os.path.join(gen_folder, 'overlap')
    os.makedirs(overlap_folder, exist_ok=True)
    data_dict['gt_path'] = os.path.join(overlap_folder, 'gt_'+se_jumpname+f"_{config['gt_voxel_size']}.bin")
    data_dict['pred_path'] = os.path.join(overlap_folder, 'pred_'+se_jumpname+'.bin')
    data_dict['tp_path'] = os.path.join(overlap_folder, 'tp_'+se_jumpname+'.bin')
    data_dict['fp_path'] = os.path.join(overlap_folder, 'fp_'+se_jumpname+'.bin')
    data_dict['fn_path'] = os.path.join(overlap_folder, 'fn_'+se_jumpname+'.bin')
    
    # os.makedirs(os.path.join(gen_folder, ''), exist_ok=True)
    # data_dict['_path'] = os.path.join(gen_folder, '', jumpname+'.')    
    
    return data_dict, config

def draw_registration_result(source, target, transformation):
    coords = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw([source.transform(transformation).paint_uniform_color([1, 0.706, 0]), target.paint_uniform_color([0, 0.651, 0.929]), coords])
    
def draw_registration_result_original_color(source, target, transformation):
    coords = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw([source.transform(transformation), target, coords])
    
class matching_result():
    def __init__(self, s, t, transformation=np.eye(4), information = np.eye(6)):
        self.s = s
        self.t = t
        self.success = False
        self.transformation = transformation
        self.information = information
    
def set_scene(input_dir):
    scene_template = os.path.join(input_dir, 'scene*')
    
    return glob.glob(scene_template)

def set_depth_scale(datatype):
    if datatype == 'ScanNet' or datatype == 'Redwood' or datatype == '3DMatch':
        depth_scale = 1000.0
    elif datatype == 'ETH3D' or datatype == 'TUM' or datatype == 'ICL-NUIM':
        depth_scale = 5000.0
    
    return depth_scale

def total_frame(datatype, scene_path):
    if datatype =='ScanNet':
        total_f = len(glob.glob(os.path.join(scene_path, 'depth','*.png')))
        
    return total_f

def gt_visualization(file, save_path):
    with open(file, 'rb') as f:
        mat = np.load(f)

    mat = np.asarray(mat*255, dtype=np.uint8)
    im = Image.fromarray(mat)
    
    im.save(save_path)
    
    return im
        
def pred_visualization(file, save_path):
    with open(file, 'rb') as f:
        mat = np.load(f)

    mask = mat < 0.0
    mat[mask] = 0.0
    mat = np.asarray(mat*255, dtype=np.uint8)
    im = Image.fromarray(mat)
    
    im.save(save_path)
    
    return im

def matrix_visualization(data_dict, config):
    gt_matrix_path = data_dict['gt_path']
    embedding_path = data_dict['embed_path']
