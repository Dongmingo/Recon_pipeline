"""
 Make global posegraph with embeddings edge: build local geometric surfaces (referred to as fragments) from short subsequences of the input RGBD sequence.
 This part uses RGBD Odometry, Multiway registration, and RGBD integration.
"""

import os
os.environ['DISPLAY'] = ':1'
import logging
import pickle

import numpy as np
import open3d as o3d

from core.optimization import run_posegraph_optimization
from utils.camera import generate_camera
from core.legacy import register_one_rgbd_pair

def embed_recon(data_dict, config):
    extract_edges_from_embed(data_dict, config)    # embeddings -> edge 저장
    make_posegraph_embed(data_dict, config)     # from edges -> posegraph 형성
    optimize_posegraph_embed(data_dict, config)     #
    integrate_rgb_frames_embed(data_dict, config)
    
    
def extract_edges_from_embed(data_dict, config):
    embed = data_dict['embeddings']
    pred_ov_mat = embed @ embed.T
    pred_mat_path = data_dict['pred_path']
    with open(pred_mat_path, 'wb')as f:
        np.save(f, pred_ov_mat)
    
    gt_path = data_dict['gt_path']
    with open(gt_path, 'rb')as f:
        gt_ov_mat = np.load(f)
    
    edge_pairs = []
    edge_path = data_dict['edge_candidate']
    overlap_th = config['overlap_th']
    pred_mask = pred_ov_mat > overlap_th
    gt_mask = gt_ov_mat > overlap_th - config['overlap_buffer']
    tp_mat = np.eye(config['n_frames'])
    fp_mat = np.zeros(config['n_frames'])
    fn_mat = np.zeros(config['n_frames'])
    for s in range(config['n_frames']-1):
        for t in range(s+1, config['n_frames']):
            if pred_mask[s,t]:
                edge_pairs.append((s,t))
                if gt_mask[s,t]:
                    tp_mat[s,t] = 1
                else:
                    fp_mat[s,t] = 1
            else:
                if gt_mask[s,t]:
                    fn_mat[s,t] = 1
                    
    with open(data_dict['tp_path'], 'rb')as f:
        np.load(f, tp_mat)
    with open(data_dict['fp_path'], 'rb')as f:
        np.load(f, fp_mat)
    with open(data_dict['fn_path'], 'rb')as f:
        np.load(f, fn_mat)

    with open(edge_path, 'wb')as f:
        pickle.dump(f, edge_pairs)
    
    
def make_posegraph_embed(data_dict, config):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    
    reg_option = config['reg_option']
    intrinsic = config['intrinsic']
    depth_scale = config['depth_scale']
    
    save_path = data_dict['legacy_posegraph_b']
    if not os.path.exists(save_path) or config['debug_mode']:
        pose_graph = o3d.pipelines.registration.PoseGraph()
        trans_odometry = np.identity(4)
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(trans_odometry))
        num_f = config['n_frames']
        success_matrix = np.eye(num_f)
        for s in range(num_f-1):
            for t in range(s + 1, num_f):
                # odometry
                if t == s + 1:
                    logging.info(
                        "RGBD matching between frame : %d and %d"
                        % (s*config['frame_jump'], t*config['frame_jump']))
                    [success, trans, info] = register_one_rgbd_pair(s, t, data_dict, intrinsic, depth_scale, reg_option)
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(s, t, trans, info, uncertain=True))
                    trans_odometry = np.dot(trans, trans_odometry)
                    trans_odometry_inv = np.linalg.inv(trans_odometry)
                    pose_graph.nodes.append(
                        o3d.pipelines.registration.PoseGraphNode(trans_odometry_inv))
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(s, t, trans, info, uncertain=False))     
                    if not success:
                        logging.info(
                            f"Odometry pair {s*config['frame_jump']} and {t*config['frame_jump']} match failed"
                        )

                # keyframe loop closure
                elif s % config['keyframe_rate'] == 0 and t % config['keyframe_rate'] == 0:
                    logging.info(
                        "RGBD matching between frame : %d and %d"
                        % (s*config['frame_jump'], t*config['frame_jump']))
                    [success, trans, info] = register_one_rgbd_pair(s, t, data_dict, intrinsic, depth_scale, reg_option)
                    if success:
                        pose_graph.edges.append(
                            o3d.pipelines.registration.PoseGraphEdge(s, t, trans, info, uncertain=True))
                    else:
                        logging.info(
                            f"Pairwise registration failed : {s*config['frame_jump']} and {t*config['frame_jump']}"
                        )
                    
                success_matrix[s, t] = success
                success_matrix[t, s] = success
        with open(data_dict['legacy_succ_reg'], 'wb')as f:
            np.save(f, success_matrix)
        logging.info(f"Pairwise Registration success matrix saved at {config['legacy_succ_reg']}") 
            
        o3d.io.write_pose_graph(save_path, pose_graph)
        logging.info(f"Posegraph before optimize saved at : {save_path}")

    
def optimize_posegraph_embed(data_dict, config):
    opt_option = config['opt_option']
    pose_graph_name = data_dict['legacy_posegraph_b']
    pose_graph_optimized_name = data_dict['legacy_posegraph_a']

    run_posegraph_optimization(pose_graph_name, pose_graph_optimized_name, opt_option)


def integrate_rgb_frames_embed(data_dict, config):
    intrinsic = config['intrinsic']
    pose_graph_name = data_dict['legacy_posegraph_a']
    
    pose_graph = o3d.io.read_pose_graph(pose_graph_name)
    edge_path = data_dict['legacy_final_edge']
    
    final_edge_mat = np.eye(config['n_frames'])
    for edge in pose_graph.edges():
        s = edge.source_node_id
        t = edge.target_node_id
        final_edge_mat[s,t], final_edge_mat[t,s] = 1,1
    with open(edge_path, 'wb')as f:
        np.save(f, final_edge_mat)
    logging.info(f"final edge matrix saved at {edge_path}")

    
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
                                voxel_length=config["tsdf_cubic_size"] / 512.0,
                                sdf_trunc=0.04,
                                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    
    for i in range(len(pose_graph.nodes)):
        logging.info(
            "integrate rgbd frame %d (of %d)." %
            ((i+1)*config['frame_jump'], len(pose_graph.nodes)*config['frame_jump']))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                                    o3d.io.read_image(data_dict['color'][i]),
                                    o3d.io.read_image(data_dict['depth'][i]),
                                    depth_scale=config['depth_scale'],
                                    convert_rgb_to_intensity=False)
        
        pose = pose_graph.nodes[i].pose
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
        
    mesh = volume.extract_triangle_mesh()
    o3d.io.write_triangle_mesh(data_dict['legacy_obj'], mesh)
    logging.info(f"final 3d reconstruction mesh saved at {data_dict['legacy_obj']}")

    mesh.compute_vertex_normals()
    
    if config['visualize']:
        start_pose = pose_graph.nodes[0].pose @ np.linalg.inv(gt_traj[0])
        
        gt_traj = data_dict['gtpose']
        gt_camera = o3d.geometry.LineSet()
        for gt_pose in gt_traj:
            gt_camera += generate_camera(start_pose @ gt_pose, [0.9, 0.1, 0.1])
            
        pred_camera = o3d.geometry.LineSet()
        for node in pose_graph.nodes:
            pred_camera += generate_camera(node.pose, [0.1, 0.1, 0.7])
        
        coords = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([coords, mesh, gt_camera, pred_camera])
        
    return mesh

            