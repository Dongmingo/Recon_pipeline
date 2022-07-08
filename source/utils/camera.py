import numpy as np
import open3d as o3d

def generate_camera(ext_mat, color= [1,0,0]):
    points, lines = cam_pos(extrinsic=ext_mat, focal_len_scaled=0.1, aspect_ratio=0.3)
    colors = [color for _ in range(8)]
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points), lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def generate_line(ext_mat_s, ext_mat_t, color= [1,0,0]):
    points = [ext_mat_s[:3,3], ext_mat_t[:3,3]]
    lines = [[0,1]]
    colors = [color]
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points), lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def cam_pos(extrinsic, focal_len_scaled=5, aspect_ratio=0.5):
    vertex_std = np.array([[0, 0, 0, 1],
                           [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                           [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                           [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                           [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]])
    vertices = vertex_std @ extrinsic.T
    lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
    return vertices[:,:3], lines