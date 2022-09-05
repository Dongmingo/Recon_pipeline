"""
Reconstruction debug module

License Info
"""

import os
os.environ['DISPLAY'] = ':1'
import glob
import time
import logging
import argparse

import open3d as o3d

from utils.utility import set_dict, set_scene, make_decision_graph
from utils.preprocess import preprocess
from core.legacy import global_fragment
from core.reconstruction import embed_recon
from core.eval import eval_embed, eval_recon

# from core.recon import reconstruction

def main(args):
    print(args)
    
    scene_list = set_scene(args.input_dir)
    
    if args.mode == 'preprocess':
        if args.multi_thread:
            from joblib import Parallel, delayed
            import multiprocessing
            import subprocess
            MAX_THREAD = min(multiprocessing.cpu_count(), len(scene_list))
            Parallel(backend='threading', n_jobs=MAX_THREAD)(delayed(preprocess)(
                                scene_path, args) for scene_path in scene_list)
        else:
            for scene_path in scene_list:
                preprocess(scene_path, args)

    
    elif args.mode == 'legacy':
        for scene_path in scene_list:
            data_dict, config = set_dict(scene_path, args)
            global_fragment(data_dict, config)
    
    elif args.mode == 'embed':
        for scene_path in scene_list:
            data_dict, config = set_dict(scene_path, args)
            
            eval_embed(data_dict, config)
            # embed_recon(data_dict, config)
            # eval_recon(data_dict, config)
            
    elif args.mode == 'decision_graph':
        make_decision_graph(scene_list, args)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # misc
    parser.add_argument('--mode', type=str, default='embed',
                        help='preprocess, legacy, embed')
    parser.add_argument('--loglevel', type=str, default='info',
                        help='debug, info, warning, error')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='Visualize during reconstruction')
    parser.add_argument('-c', '--camera', action='store_true',
                        help='Visualize camera pose')
    parser.add_argument('-g', '--gpu', action='store_true',
                        help='Using GPU, tensor based reconstruction')
    parser.add_argument('-o', '--output', action='store_true',
                        help='Set to save results at other location')
    parser.add_argument('-m', '--multi-thread', action='store_true',
                        help='Set for Multi-threading')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='DEBUGGING')
    
    # dataset, result path
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing reconstruction data')
    parser.add_argument('--output_dir', type=str, default='results/')

    # preprocess config
    parser.add_argument('--gt_voxel_size', type=float, default=0.05,
                        help='Minkowski Quantize gt voxel_size')
    
    # registration config
    parser.add_argument('--datatype', type=str, default="ScanNet",
                        help='scanNet')
    parser.add_argument('--frame_jump', type=int, default=30,
                        help='frame jump, default 30')
    parser.add_argument('--keyframe_rate', type=int, default=1,
                        help='keyframe_rate after frame jump, default 1')
    parser.add_argument('--start_index', type=int, default=0,
                        help='start frame index, default 0')
    parser.add_argument('--end_index', type=int, default=0,
                        help='end frame index, default(len(total_f)-1)')
    parser.add_argument('--voxel_size', type=float, default=0.01,
                        help='reconstruction resolution')
    parser.add_argument('--depth_trunc', type=float, default=5.0,
                        help='maximum depth range default=5.0m')
    parser.add_argument('--prone_th', type=float, default=0.25,
                        help='posegraph optimization prone threshold (0.0, 1.0)')
    parser.add_argument('--tsdf_cubic_size', type=float, default=4.0,
                        help='tsdf resolution')
    parser.add_argument('--py_c', type=int, default=30,
                        help='registration pyramid level coarse')
    parser.add_argument('--py_m', type=int, default=10,
                        help='registration pyramid level medium')
    parser.add_argument('--py_f', type=int, default=5,
                        help='registration pyramid level fine')
    parser.add_argument('--icp_method', type=str, default='color',
                        help='point_to_point, point_to_plane, color(default), generalized')
    parser.add_argument('--global_registration', type=str, default='fgr',
                        help='ransac, fgr')
    
    # recon use embedding
    parser.add_argument('--overlap_th', type=float, default=0.7,
                        help='embeddings cosine similarity threshold to mine candidate, default 0.7')
    parser.add_argument('--gt_overlap_th', type=float, default=0.3,
                        help='ground truth overlap boundary, default 0.3')
    parser.add_argument('--bin_size', type=float, default=0.05,
                        help='bin size for making Decision Graph , default 0.05')
    parser.add_argument('--weird_gap', type=float, default=0.5,
                        help='weird pair mining gap for |gt - pred|, default 0.5')
    parser.add_argument('--good_gap', type=float, default=0.05,
                        help='weird pair mining gap for |gt - pred|, default 0.05')
    

    args = parser.parse_args()
    
    tm = time.localtime(time.time())
    exetime = str('{:02d}'.format(tm.tm_year%100))+str('{:02d}'.format(tm.tm_mon))+str('{:02d}'.format(tm.tm_mday))+'_' \
                +str('{:02d}'.format(tm.tm_hour))+str('{:02d}'.format(tm.tm_min))
    os.makedirs('log', exist_ok=True)
    log_path = os.path.join('log', exetime+'.log')
    
    logger = logging.getLogger()  
    logger.setLevel(getattr(logging, args.loglevel.upper()))
    fh = logging.FileHandler(log_path)
    sh = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s : %(name)s  : %(funcName)s : %(levelname)s : %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    
    main(args)