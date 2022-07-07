import open3d as o3d
import logging

def run_posegraph_optimization(pose_graph_name, pose_graph_optimized_name, opt_option):
    
    pose_graph = o3d.io.read_pose_graph(pose_graph_name)
    
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(pose_graph,
                                                        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                                                        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                                                        opt_option)
        
    o3d.io.write_pose_graph(pose_graph_optimized_name, pose_graph)
    logging.info(f"Optimized posegraph saved at {pose_graph_optimized_name}")

    return pose_graph