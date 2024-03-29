"""
Utilities
"""
import os
os.environ['DISPLAY'] = ':1'
import glob
import logging
import pickle

import numpy as np
import open3d as o3d

from PIL import Image
import matplotlib.pyplot as plt

from utils.load_data import load_intrinsic, load_image_size, load_extrinsic


def set_dict(scene, args):
    input_dir = scene
    logging.info(f"Setting Configuration for reconstruct scene : {input_dir}")    
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
    config["end_index"] = args.end_index if args.end_index != 0 and args.end_index < total_f-1 else total_f-1
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
    config['gt_overlap_th'] = args.gt_overlap_th
    config['bin_size'] = args.bin_size
    config['weird_gap'] = args.weird_gap
    config['good_gap'] = args.good_gap
    
    logging.info("Setting data path")
    
    data_dict = {
        'index' : [],
        'color' : [],
        'depth' : [],
        'gtpose' : [],
        'gt_pcd_v' : [],
        'no_gt_info' : [],
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
            gt_pose = load_extrinsic(gt_path, config['datatype'], index)
            data_dict['gtpose'].append(gt_pose)
            if -np.inf in gt_pose:
                data_dict['no_gt_info'].append(index//config['frame_jump'])
    
    if args.output:
        data_dict['output'] = args.output_dir
    else:
        output_folder = os.path.join(gen_folder, 'results')
        os.makedirs(output_folder, exist_ok=True)
        
        data_dict['output'] = output_folder
        
    if not args.mode == 'preprocess':
        data_dict['embed_path'] = os.path.join(gen_folder, 'embeddings.bin')
        with open(data_dict['embed_path'], 'rb')as f:
            embed_t = np.load(f)
        data_dict['embeddings'] = embed_t[data_dict['index']]
    
    se_jumpname = f"{config['start_index']}-{config['end_index']}_{config['frame_jump']}"
    ov_se_jumpname = se_jumpname+f"_{config['overlap_th']}"
    ov_bin_se_jumpname = se_jumpname+f"_{config['gt_overlap_th']}"+f"_{config['bin_size']}"
    
    obj_folder = os.path.join(gen_folder, 'obj')
    os.makedirs(obj_folder, exist_ok=True)
    data_dict['legacy_obj'] = os.path.join(obj_folder, 'legacy_'+se_jumpname+'.obj')
    data_dict['embed_obj'] = os.path.join(obj_folder, 'embed_'+ ov_se_jumpname + ".obj")
                          
    ply_folder =os.path.join(gen_folder, 'ply')         
    os.makedirs(ply_folder, exist_ok=True)
    data_dict['ply_template'] = os.path.join(ply_folder, se_jumpname+'.ply') 
    
    edge_folder = os.path.join(gen_folder, 'edge')
    os.makedirs(os.path.join(edge_folder), exist_ok=True)
    data_dict['legacy_succ_edge'] = os.path.join(edge_folder, 'legacy_success_'+se_jumpname+'.bin')
    data_dict['legacy_final_edge'] = os.path.join(edge_folder, 'legacy_final_'+se_jumpname+'.bin')
    data_dict['embed_succ_edge'] = os.path.join(edge_folder, 'embed_success_'+ ov_se_jumpname + ".bin")
    data_dict['embed_final_edge'] = os.path.join(edge_folder, 'embed_final_'+ ov_se_jumpname + ".bin")
    data_dict['edge_candidate'] = os.path.join(edge_folder, 'edge_candidate_'+ ov_se_jumpname + ".pickle")
    
    reg_eval_folder = os.path.join(gen_folder, 'reg_eval')
    os.makedirs(reg_eval_folder, exist_ok=True)
    data_dict['gt_pred_vis'] = os.path.join(reg_eval_folder, 'gt_pred_'+se_jumpname+'.png')
    data_dict['tpfpfntn_vis'] = os.path.join(reg_eval_folder, 'tpfpfntn_'+ov_se_jumpname+'.png')
    data_dict['succ_fin_vis'] = os.path.join(reg_eval_folder, 'succ_fin_'+ov_se_jumpname+'.png')
    data_dict['weird_pairs'] = os.path.join(reg_eval_folder, 'weird_edges_'+ se_jumpname +'_'+ str(config['weird_gap'])+".pickle")
    data_dict['good_pairs'] = os.path.join(reg_eval_folder, 'good_edges_'+ se_jumpname +'_'+ str(config['good_gap'])+".pickle")
    data_dict['bins_save_path'] = os.path.join(reg_eval_folder, 'bins_'+ ov_bin_se_jumpname + ".pickle")
    os.makedirs(os.path.join(reg_eval_folder, 'weird_vis'), exist_ok=True)
    data_dict['weird_vis_template'] = reg_eval_folder+'/weird_vis/%05d-%05d.png'
    
    os.makedirs(os.path.join(gen_folder, 'posegraph'), exist_ok=True)
    data_dict['legacy_posegraph_b'] = os.path.join(gen_folder, 'posegraph', 'legacy_b_'+se_jumpname+'.json')  
    data_dict['legacy_posegraph_a'] = os.path.join(gen_folder, 'posegraph', 'legacy_a_'+se_jumpname+'.json')  
    data_dict['embed_posegraph_b'] = os.path.join(gen_folder, 'posegraph', 'embed_b_'+ov_se_jumpname+'.json')  
    data_dict['embed_posegraph_a'] = os.path.join(gen_folder, 'posegraph', 'embed_a_'+ov_se_jumpname+'.json')  
    
    overlap_folder = os.path.join(gen_folder, 'overlap')
    os.makedirs(overlap_folder, exist_ok=True)
    data_dict['gt_path'] = os.path.join(overlap_folder, 'gt_'+se_jumpname+f"_{config['gt_voxel_size']}.bin")
    data_dict['pred_path'] = os.path.join(overlap_folder, 'pred_'+se_jumpname+'.bin')
    data_dict['tp_path'] = os.path.join(overlap_folder, 'tp_'+ ov_se_jumpname + ".bin")
    data_dict['fp_path'] = os.path.join(overlap_folder, 'fp_'+ ov_se_jumpname + ".bin")
    data_dict['fn_path'] = os.path.join(overlap_folder, 'fn_'+ ov_se_jumpname + ".bin")
    
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

def make_decision_graph(scene_list, args):
    total_diff = 0
    bin_size = args.bin_size
    bins = np.arange(0.0, 1.0, bin_size)
    
    total_dict = {'num_total_pairs' : 0,
                  'num_ignore_pairs' : 0,
                  'tp_bins': np.zeros(len(bins)),
                  'fp_bins': np.zeros(len(bins)),
                  'fn_bins': np.zeros(len(bins))}
    
    for scene_path in scene_list:
        bins_path = glob.glob(os.path.join(scene_path, 'generated_data/reg_eval/bins_')+f"*-*_{args.frame_jump}_{args.gt_overlap_th}_{args.bin_size}.pickle")[0]
        assert os.path.exists(bins_path), f"{bins_path} not exists"
        
        with open(bins_path, 'rb')as f:
            bins_dict = pickle.load(f)
            
        total_dict['num_total_pairs'] += bins_dict['num_total_pairs']
        total_dict['num_ignore_pairs'] += bins_dict['num_ignore_pairs']
        total_dict['tp_bins'] += bins_dict['tp_bins']
        total_dict['fp_bins'] += bins_dict['fp_bins']
        total_dict['fn_bins'] += bins_dict['fn_bins']
        total_diff += bins_dict['num_total_pairs'] * bins_dict['avg_diff']
    
    total_dict['avg_diff'] = total_diff / total_dict['num_total_pairs']
    total_dict['recall_bins'] = total_dict['tp_bins'] / (total_dict['tp_bins'] + total_dict['fn_bins'])
    total_dict['precision_bins'] = total_dict['tp_bins'] / (total_dict['tp_bins'] + total_dict['fp_bins'])
    recall = total_dict['recall_bins']
    precision = total_dict['precision_bins']
    logging.info(f"TP bins :: {' '.join(map(str,total_dict['tp_bins']))}")
    logging.info(f"FP bins :: {' '.join(map(str,total_dict['fp_bins']))}")
    logging.info(f"FN bins :: {' '.join(map(str,total_dict['fn_bins']))}")
    
    
    fig = plt.figure(figsize=(12,9))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.xlim([0.0, 1.0])
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.title(f"PR curve at {args.gt_overlap_th}, scene : {len(scene_list)}, total_pairs : {total_dict['num_total_pairs']}, avg_diff : {total_dict['avg_diff']}")
    if args.visualize:
        plt.show()
    plt.savefig(f"PR_curve_{args.gt_overlap_th}.png", dpi = 300)
    plt.clf()
        