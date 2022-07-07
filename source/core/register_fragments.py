"""
 Register fragments: the fragments are aligned in a global space to detect loop closure. 
 This part uses Global registration, ICP registration, and Multiway registration.
"""

import os

import numpy as np
import open3d as o3d

from optimization import run_posegraph_optimization
from refine_registration import multiscale_icp
from utils.utility import draw_registration_result, matching_result


def preprocess_point_cloud(pcd, config):
    voxel_size = config["voxel_size"]
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                             max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0,
                                             max_nn=100))
    return (pcd_down, pcd_fpfh)

def compute_initial_registration(s, t, source_down, target_down, source_fpfh,
                                 target_fpfh, data_dict, config):

    if t == s + 1:  # odometry case
        print("Using RGBD odometry")
        pose_graph_frag = o3d.io.read_pose_graph(
            data_dict['template_fragment_posegraph_optimized'] % s)
        n_nodes = len(pose_graph_frag.nodes)
        transformation_init = np.linalg.inv(pose_graph_frag.nodes[n_nodes -
                                                                  1].pose)
        (transformation, information) = \
                multiscale_icp(source_down, target_down,
                [config["voxel_size"]], [50], config, transformation_init)
    else:  # loop closure case
        (success, transformation,
         information) = register_point_cloud_fpfh(source_down, target_down,
                                                  source_fpfh, target_fpfh,
                                                  config)
        if not success:
            print("No reasonable solution. Skip this pair")
            return (False, np.identity(4), np.zeros((6, 6)))
    print(transformation)

    if config["debug_mode"]:
        draw_registration_result(source_down, target_down, transformation)
    return (True, transformation, information)


def register_point_cloud_pair(data_dict, s, t, config):
    source_pcd = o3d.io.read_point_cloud(data_dict['frag_pcd'][s])
    target_pcd = o3d.io.read_point_cloud(data_dict['frag_pcd'][t])
    (source_down, source_fpfh) = preprocess_point_cloud(source_pcd, config)
    (target_down, target_fpfh) = preprocess_point_cloud(target_pcd, config)

    results = compute_initial_registration(s, t, source_down, target_down, source_fpfh, target_fpfh, data_dict, config)
    
    return results
    

def register_point_cloud_fpfh(source, target, source_fpfh, target_fpfh, config):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    distance_threshold = config["voxel_size"] * 1.4
    if config["global_registration"] == "fgr":
        result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
    if config["global_registration"] == "ransac":
        # Fallback to preset parameters that works better
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh, False, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(
                False), 4,
            [
                o3d.pipelines.registration.
                CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(
                1000000, 0.999))
    if (result.transformation.trace() == 4.0):
        return (False, np.identity(4), np.zeros((6, 6)))
    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, distance_threshold, result.transformation)
    if information[5, 5] / min(len(source.points), len(target.points)) < 0.3:
        return (False, np.identity(4), np.zeros((6, 6)))
    return (True, result.transformation, information)


def update_posegraph_for_scene(s, t, transformation, information, odometry,
                               pose_graph):
    if t == s + 1:  # odometry case
        odometry = np.dot(transformation, odometry)
        odometry_inv = np.linalg.inv(odometry)
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(odometry_inv))
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(s,
                                                     t,
                                                     transformation,
                                                     information,
                                                     uncertain=False))
    else:  # loop closure case
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(s,
                                                     t,
                                                     transformation,
                                                     information,
                                                     uncertain=True))
    return (odometry, pose_graph)


def optimize_posegraph_for_scene(data_dict, config):
    opt_option = config['opt_option']
    pose_graph_name = data_dict['template_initial_posegraph']
    pose_graph_optimized_name = data_dict['template_registered_posegraph']
    run_posegraph_optimization(pose_graph_name, pose_graph_optimized_name, opt_option)
    
    
def make_posegraph_for_scene(data_dict, config):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    n_files = config['n_fragments']
    matching_results = {}
    for s in range(n_files):
        for t in range(s + 1, n_files):
            matching_results[s * n_files + t] = matching_result(s, t)

    if config["python_multi_threading"] == True:
        from joblib import Parallel, delayed
        import multiprocessing
        import subprocess
        MAX_THREAD = min(multiprocessing.cpu_count(),
                         max(len(matching_results), 1))
        results = Parallel(n_jobs=MAX_THREAD)(delayed(
            register_point_cloud_pair)(data_dict, matching_results[r].s,
                                       matching_results[r].t, config)
                                              for r in matching_results)
        for i, r in enumerate(matching_results):
            matching_results[r].success = results[i][0]
            matching_results[r].transformation = results[i][1]
            matching_results[r].information = results[i][2]
    else:
        for r in matching_results:
            (matching_results[r].success, matching_results[r].transformation,
                    matching_results[r].information) = \
                    register_point_cloud_pair(data_dict,
                    matching_results[r].s, matching_results[r].t, config)

    for r in matching_results:
        if matching_results[r].success:
            (odometry, pose_graph) = update_posegraph_for_scene(
                matching_results[r].s, matching_results[r].t,
                matching_results[r].transformation,
                matching_results[r].information, odometry, pose_graph)
    o3d.io.write_pose_graph(
        data_dict['template_registered_posegraph'],
        pose_graph)
    
    
def register_fragments(data_dict, config):
    
    make_posegraph_for_scene(data_dict, config)
    optimize_posegraph_for_scene(data_dict, config)


