import os
os.environ['DISPLAY'] = ':1'
import pickle
import logging

import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt
import matplotlib.image as img
from utils.camera import generate_camera, generate_line


def eval_embed(data_dict, config):
    extract_edges_from_embed(data_dict, config)
    # vis_gt_pred(data_dict, config)
    # vis_tpfpfntn(data_dict, config)
    # vis_weird_pairs(data_dict, config)
    
def vis_weird_pairs(data_dict, config):
    weird_path = data_dict['weird_pairs']
    with open(weird_path, 'rb')as f:
        weird_pairs = pickle.load(f)
    
    logging.info(f"Total weird pairs {len(weird_pairs)} with gap : {config['weird_gap']}")
    for (s, t, gt, pred) in weird_pairs:
        save_path = data_dict['weird_vis_template'] % (s, t)
        color_s = img.imread(data_dict['color'][s])
        color_t = img.imread(data_dict['color'][t])
        depth_s = img.imread(data_dict['depth'][s])
        depth_t = img.imread(data_dict['depth'][t])
        
        fig = plt.figure(figsize=(12,9))
        plt.suptitle(f"weird {s} - {t} pair, gt {gt}, pred {pred}")
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        ax1.title.set_text(f"color_{s}")
        ax1.imshow(color_s)
        ax2.title.set_text(f"color_{t}")
        ax2.imshow(color_t)
        ax3.title.set_text(f"depth_{s}")
        ax3.imshow(depth_s)
        ax4.title.set_text(f"depth_{t}")
        ax4.imshow(depth_t)
        if config['visualize']:
            plt.show()
        plt.savefig(save_path, dpi = 300)
        plt.clf()


def vis_gt_pred(data_dict, config):
    trash_nodes = data_dict['no_gt_info']
    n_frames = config['n_frames']
    
    trash_mask = np.zeros((n_frames, n_frames))
    trash_mask[:, trash_nodes] = 1
    trash_mask[trash_nodes, :] = 1
    trash_mask = np.asarray(trash_mask, dtype=bool)
    gt_pred_path = data_dict['gt_pred_vis']
    with open(data_dict['gt_path'], 'rb') as f:
        gt_mat = np.load(f)
    with open(data_dict['pred_path'], 'rb') as f:
        pred_mat = np.load(f)
    diff_mat = np.abs(pred_mat - gt_mat)
    weird_pairs = []
    good_pairs = []
    for s in range(n_frames-1):
        for t in range(s+1, n_frames):            
            if not (s in trash_nodes and t in trash_nodes):
                if diff_mat[s,t] > config['weird_gap']:
                    weird_pairs.append((s,t, gt_mat[s, t], pred_mat[s,t]))
                elif diff_mat[s,t] < config['good_gap']:
                    good_pairs.append((s,t, gt_mat[s, t], pred_mat[s,t]))
    
    with open(data_dict['weird_pairs'], 'wb') as f:
        pickle.dump(weird_pairs, f)
    
    with open(data_dict['good_pairs'], 'wb')as f:
        pickle.dump(good_pairs, f)
    
    gt_mat[trash_mask] = 1
    pred_mat[trash_mask] = 1
    
    fig = plt.figure(figsize=(20,7))
    plt.suptitle(f"{os.path.split(data_dict['scene_path'])[1]}")
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.title.set_text('gt_overlap')
    ax1.imshow(1-gt_mat)
    ax2.title.set_text('pred_overlap')
    ax2.imshow(1-pred_mat)
    ax3.title.set_text('abs_diff')
    ax3.imshow(1-diff_mat)
    if config['visualize']:
        plt.show()
    plt.savefig(gt_pred_path, dpi = 300)
    plt.clf()


def vis_tpfpfntn(data_dict, config):
    trash_nodes = data_dict['no_gt_info']
    n_frames = config['n_frames']
    
    tpfpfntn_path = data_dict['tpfpfntn_vis']
    pure_mat = np.ones((n_frames, n_frames, 3))
    trash_mask = np.zeros((n_frames, n_frames, 3))
    trash_mask[:, trash_nodes, :] = 1
    trash_mask[trash_nodes, :, :] = 1
    trash_mask = np.array(trash_mask, dtype=bool)
    
    tn_mat = np.ones((n_frames, n_frames))
    with open(data_dict['tp_path'], 'rb') as f:
        tp_mat = np.load(f)
    blue = [0,0,1]
    tp_mask = np.stack([tp_mat*blue[0], tp_mat*blue[1], tp_mat*blue[2]], axis=2)
    tn_mat[np.array(tp_mat, dtype=bool)] = 0
    
    with open(data_dict['fp_path'], 'rb') as f:
        fp_mat = np.load(f)
    bluelike = [0.25,0,0.5]
    fp_mask = np.stack([fp_mat*bluelike[0], fp_mat*bluelike[1], fp_mat*bluelike[2]], axis=2)
    tn_mat[np.array(fp_mat, dtype=bool)] = 0
    
    with open(data_dict['fn_path'], 'rb') as f:
        fn_mat = np.load(f)
    redlike = [0.5,0,0.25]
    fn_mask = np.stack([fn_mat*redlike[0], fn_mat*redlike[1], fn_mat*redlike[2]], axis=2)
    tn_mat[np.array(fn_mat, dtype=bool)] = 0
    
    red = [1,0,0]
    tn_mask = np.stack([tn_mat*red[0], tn_mat*red[1], tn_mat*red[2]], axis=2)
    
    pure_mat = tp_mask + fp_mask + fn_mask + tn_mask
    pure_mat[trash_mask] = 0
    plt.imshow(pure_mat)
    plt.title(f"{os.path.split(data_dict['scene_path'])[1]}")
    if config['visualize']:
        plt.show()
    plt.savefig(tpfpfntn_path, dpi = 300)
    plt.clf()
    
    
def extract_edges_from_embed(data_dict, config):
    embed = data_dict['embeddings']
    pred_ov_mat = embed @ embed.T
    pred_ov_mat[pred_ov_mat<0] = 0
    pred_mat_path = data_dict['pred_path']
    with open(pred_mat_path, 'wb')as f:
        np.save(f, pred_ov_mat)
    
    gt_path = data_dict['gt_path']
    with open(gt_path, 'rb')as f:
        gt_ov_mat = np.load(f)
        
    n = config['n_frames'] - len(data_dict['no_gt_info'])
    total = n * (n-1) / 2
    ig = config['n_frames'] * (config['n_frames']-1) / 2 - total
    if len(data_dict['no_gt_info']) != 0:
        pred_ov_mat[np.array(data_dict['no_gt_info']),:] = -1.0
        pred_ov_mat[:,np.array(data_dict['no_gt_info'])] = -1.0
    mask = (gt_ov_mat <1).astype(int) * (pred_ov_mat >= 0).astype(int)
    abs_diff = np.abs(pred_ov_mat - gt_ov_mat)
    avg_diff = np.sum(abs_diff[mask==1]) / (total * 2)
    
    bin_size = config['bin_size']
    bins = np.arange(0.0, 1.0, bin_size)
    tp_bins, fp_bins, fn_bins = np.zeros(len(bins)), np.zeros(len(bins)), np.zeros(len(bins))
    
    edge_pairs = []
    edge_path = data_dict['edge_candidate']
    overlap_th = config['overlap_th']
    pred_mask = pred_ov_mat >= overlap_th
    gt_mask = gt_ov_mat >= config['gt_overlap_th']
    tp_mat = np.eye(config['n_frames'])
    fp_mat = np.zeros((config['n_frames'],config['n_frames']))
    fn_mat = np.zeros((config['n_frames'],config['n_frames']))
    for s in range(config['n_frames']-1):
        for t in range(s+1, config['n_frames']):
            if s not in data_dict['no_gt_info'] and t not in data_dict['no_gt_info']:
                th_index = int(pred_ov_mat[s,t]/bin_size)+1
                if pred_ov_mat[s,t] ==1:
                    th_index = 20
                if gt_ov_mat[s,t] >= config['gt_overlap_th']:
                    tp_bins += np.hstack((np.ones(th_index),np.zeros(len(bins)-th_index)))
                    fn_bins += np.hstack((np.zeros(th_index),np.ones(len(bins)-th_index)))
                else:
                    fp_bins += np.hstack((np.ones(th_index),np.zeros(len(bins)-th_index)))
            if pred_mask[s,t]:
                edge_pairs.append((s,t))
                if gt_mask[s,t]:
                    tp_mat[s,t], tp_mat[t,s] = 1, 1
                else:
                    fp_mat[s,t], fp_mat[t,s] = 1, 1
            else:
                if gt_mask[s,t]:
                    fn_mat[s,t], fn_mat[t,s] = 1, 1
    
    recall_bins = tp_bins / (tp_bins + fn_bins)
    precision_bins = tp_bins / (tp_bins + fp_bins) 
        
    bins_dict = {'scene_path' : data_dict['scene_path'],
                  'num_total_pairs' : total,
                  'num_ignore_pairs' : ig,
                  'avg_diff' : avg_diff,
                  'tp_bins': tp_bins,
                  'fp_bins': fp_bins,
                  'fn_bins': fn_bins,
                  'recall_bins': recall_bins,
                  'precision_bins': precision_bins}
    
    with open(data_dict['bins_save_path'], 'wb')as f:
        pickle.dump(bins_dict, f)
                    
    with open(data_dict['tp_path'], 'wb')as f:
        np.save(f, tp_mat)
    with open(data_dict['fp_path'], 'wb')as f:
        np.save(f, fp_mat)
    with open(data_dict['fn_path'], 'wb')as f:
        np.save(f, fn_mat)

    with open(edge_path, 'wb')as f:
        pickle.dump(edge_pairs, f)
        

def eval_recon(data_dict, config):
    coords = o3d.geometry.TriangleMesh.create_coordinate_frame()
    intrinsic = config['intrinsic']
    posegraph_b_path = data_dict['embed_posegraph_b']
    posegraph_a_path = data_dict['embed_posegraph_a']
    gt_traj = data_dict['gtpose']
    
    tp_edges, fp_edges, fn_edges = [], [], []
    with open(data_dict['tp_path'], 'rb') as f:
        tp_mat = np.load(f)
    with open(data_dict['fp_path'], 'rb') as f:
        fp_mat = np.load(f)
    with open(data_dict['fn_path'], 'rb') as f:
        fn_mat = np.load(f)    
    sum_mat = tp_mat + 2*fp_mat + 4*fn_mat
    for s in range(config['n_frames']-1):
        for t in range(s+2, config['n_frames']):
            val = sum_mat[s, t]
            if val == 1:
                tp_edges.append((s,t))
            elif val == 2:
                fp_edges.append((s,t))
            elif val == 4:
                fn_edges.append((s,t))    
    
    if config['visualize']:
        pose_graph_b = o3d.io.read_pose_graph(posegraph_b_path)
        volume_b = o3d.pipelines.integration.ScalableTSDFVolume(
                                voxel_length=config["tsdf_cubic_size"] / 512.0,
                                sdf_trunc=0.04,
                                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    
        for i in range(len(pose_graph_b.nodes)):
            logging.info(
                "integrate rgbd frame %d (of %d)." %
                ((i)*config['frame_jump'], (len(pose_graph_b.nodes)-1)*config['frame_jump']))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                                        o3d.io.read_image(data_dict['color'][i]),
                                        o3d.io.read_image(data_dict['depth'][i]),
                                        depth_scale=config['depth_scale'],
                                        convert_rgb_to_intensity=False)
            pose = pose_graph_b.nodes[i].pose
            volume_b.integrate(rgbd, intrinsic, np.linalg.inv(pose))
            
        mesh_b = volume_b.extract_triangle_mesh()
        
        start_pose = pose_graph_b.nodes[0].pose @ np.linalg.inv(gt_traj[0])
        gt_camera_b = o3d.geometry.LineSet()
        for gt_pose in gt_traj:
            gt_camera_b += generate_camera(start_pose @ gt_pose, [0.9, 0.1, 0.1])
            
        pred_camera_b = o3d.geometry.LineSet()
        for node in pose_graph_b.nodes:
            pred_camera_b += generate_camera(node.pose, [0.1, 0.1, 0.7])
            
        odo_edge_line = o3d.geometry.LineSet()
        for s_idx in range(config['n_frames']-2):
            odo_edge_line += generate_line(pose_graph_b.nodes[s_idx].pose, pose_graph_b.nodes[s_idx+1].pose, [0.,0.,0.])
            
        tp_edge_line = o3d.geometry.LineSet()
        for (s_idx, t_idx) in tp_edges:
            tp_edge_line += generate_line(pose_graph_b.nodes[s_idx].pose, pose_graph_b.nodes[t_idx].pose, [0.,0.,1.])
            
        fp_edge_line = o3d.geometry.LineSet()
        for (s_idx, t_idx) in fp_edges:
            fp_edge_line += generate_line(pose_graph_b.nodes[s_idx].pose, pose_graph_b.nodes[t_idx].pose, [0.25,0,0.5])
        
        fn_edge_line = o3d.geometry.LineSet()
        for (s_idx, t_idx) in fn_edges:
            fn_edge_line += generate_line(pose_graph_b.nodes[s_idx].pose, pose_graph_b.nodes[t_idx].pose, [0.5,0,0.25])
            
        o3d.visualization.draw_geometries([coords, mesh_b, gt_camera_b, pred_camera_b, odo_edge_line, tp_edge_line, fp_edge_line, fn_edge_line])
            
        pose_graph_a = o3d.io.read_pose_graph(posegraph_a_path)
        volume_a = o3d.pipelines.integration.ScalableTSDFVolume(
                                voxel_length=config["tsdf_cubic_size"] / 512.0,
                                sdf_trunc=0.04,
                                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        
        for j in range(len(pose_graph_a.nodes)):
            logging.info(
                "integrate rgbd frame %d (of %d)." %
                ((j)*config['frame_jump'], (len(pose_graph_a.nodes)-1)*config['frame_jump']))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                                        o3d.io.read_image(data_dict['color'][j]),
                                        o3d.io.read_image(data_dict['depth'][j]),
                                        depth_scale=config['depth_scale'],
                                        convert_rgb_to_intensity=False)
            pose = pose_graph_a.nodes[j].pose
            volume_a.integrate(rgbd, intrinsic, np.linalg.inv(pose))
            
        mesh_a = volume_a.extract_triangle_mesh()
        
        start_pose = pose_graph_a.nodes[0].pose @ np.linalg.inv(gt_traj[0])
        gt_camera_a = o3d.geometry.LineSet()
        for gt_pose in gt_traj:
            gt_camera_a += generate_camera(start_pose @ gt_pose, [0.9, 0.1, 0.1])
            
        pred_camera_a = o3d.geometry.LineSet()
        for node in pose_graph_a.nodes:
            pred_camera_a += generate_camera(node.pose, [0.1, 0.1, 0.7])
            
        # visualize edges
        odo_edge_line = o3d.geometry.LineSet()
        for s_idx in range(config['n_frames']-2):
            odo_edge_line += generate_line(pose_graph_a.nodes[s_idx].pose, pose_graph_a.nodes[s_idx+1].pose, [0.0,0.0,0.0])
            
        tp_edge_line = o3d.geometry.LineSet()
        for (s_idx, t_idx) in tp_edges:
            tp_edge_line += generate_line(pose_graph_a.nodes[s_idx].pose, pose_graph_a.nodes[t_idx].pose, [0.,0.,1.])
            
        fp_edge_line = o3d.geometry.LineSet()
        for (s_idx, t_idx) in fp_edges:
            fp_edge_line += generate_line(pose_graph_a.nodes[s_idx].pose, pose_graph_a.nodes[t_idx].pose, [0.25,0,0.5])
        
        fn_edge_line = o3d.geometry.LineSet()
        for (s_idx, t_idx) in fn_edges:
            fn_edge_line += generate_line(pose_graph_a.nodes[s_idx].pose, pose_graph_a.nodes[t_idx].pose, [0.5,0,0.25])
        
        o3d.visualization.draw_geometries([coords, mesh_a, gt_camera_a, pred_camera_a, odo_edge_line, tp_edge_line, fp_edge_line])