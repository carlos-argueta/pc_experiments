import uuid
import numpy as np
import quaternion

from copy import deepcopy

from ultralytics import YOLO
import open3d as o3d
from scipy.spatial import Delaunay

# Back-project 2D to 3D
def back_project_2D_to_3D(x, y, depth, K, R=None, t=None):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    X_cam, Y_cam, Z_cam = (x - cx) * depth / fx, (y - cy) * depth / fy, depth
    P_cam = np.array([X_cam, Y_cam, Z_cam])

    if R is not None and t is not None:
        return np.dot(R.T, (P_cam - t))

    return P_cam

# Convert 2D bounding box to 3D frustum points
def bbox_2D_to_frustum(bbox_2D, K, z_min, z_max, R=None, t=None, expand_ratio=0.0):
    x1, y1, x2, y2 = bbox_2D

    # Update the bounding box corners based on the expand_ratio
    x1 -= (x2 - x1) * expand_ratio / 2
    y1 -= (y2 - y1) * expand_ratio / 2
    x2 += (x2 - x1) * expand_ratio / 2
    y2 += (y2 - y1) * expand_ratio / 2

    # Back-project the 2D bounding box corners to 3D points
    P1_zmin = back_project_2D_to_3D(x1, y1, z_min, K, R, t)
    P2_zmin = back_project_2D_to_3D(x2, y1, z_min, K, R, t)
    P3_zmin = back_project_2D_to_3D(x2, y2, z_min, K, R, t)
    P4_zmin = back_project_2D_to_3D(x1, y2, z_min, K, R, t)

    P1_zmax = back_project_2D_to_3D(x1, y1, z_max, K, R, t)
    P2_zmax = back_project_2D_to_3D(x2, y1, z_max, K, R, t)
    P3_zmax = back_project_2D_to_3D(x2, y2, z_max, K, R, t)
    P4_zmax = back_project_2D_to_3D(x1, y2, z_max, K, R, t)

    return np.array([P1_zmin, P2_zmin, P3_zmin, P4_zmin, P1_zmax, P2_zmax, P3_zmax, P4_zmax])

# Check if a point is inside the frustum
def is_point_inside_frustum(point, delaunay):
    return delaunay.find_simplex(point) >= 0

# Filter the points inside the frustum
def filter_points_inside_frustum(point_cloud, frustum_points):
    delaunay = Delaunay(frustum_points)
    return np.array([point for point in point_cloud if is_point_inside_frustum(point, delaunay)])

# Load Yolo V8 model
model = YOLO("yolov8n.pt")

# Specify paths
date_time_str = "lidar_2022_05_05_16_04_09.0598"
image_src = f"../pcds/{date_time_str}.jpg"
cloud_src = f"../pcds/{date_time_str}.pcd"

# Inference on an image
results = model.predict(source=image_src, show=True) 

# Load and visualize point cloud
cloud = o3d.io.read_point_cloud(cloud_src)
cloud_copy = deepcopy(cloud)

num_original_points = len(cloud.points)
black_colors = np.zeros((num_original_points, 3))
cloud.colors = o3d.utility.Vector3dVector(black_colors)

# Define Camera parameters
K = np.array([[261.6734924316406, 0.0, 342.7066345214844], [0.0, 261.6734924316406, 186.6942596435547], [0.0, 0.0, 1.0]])
R = quaternion.as_rotation_matrix(quaternion.from_float_array([0.5, 0.5, -0.5, 0.5]))
t = np.array([0.087, 0.060, -0.076])     

bbox_2Ds = results[0].boxes.xyxy.cpu().numpy()               
labels = results[0].boxes.cls.cpu().numpy()                  
z_min, z_max = 0.5, 50   

z_max_range = np.arange(z_min, z_max, 0.5)

all_frustum_points = []
max_frustum_points = []

for i, bbox_2D in enumerate(bbox_2Ds):
    for z_max_i in z_max_range:
        frustum_points = bbox_2D_to_frustum(bbox_2D, K, z_min, z_max_i, R, t, expand_ratio=1.0)
        all_frustum_points.append((labels[i], frustum_points))
        if z_max_i == z_max_range[-1]:
            max_frustum_points.append((labels[i], frustum_points))

# VISUALIZE FRUSTUMS
num_labels = 80
label_colors = [np.random.rand(3) for _ in range(num_labels)]

for label, frustum_points in all_frustum_points:
    new_cloud = o3d.geometry.PointCloud()
    new_cloud.points = o3d.utility.Vector3dVector(frustum_points)
    new_cloud.colors = o3d.utility.Vector3dVector(np.full((len(frustum_points), 3), label_colors[int(label)]))
    cloud += new_cloud

# Create a custom visualization window with a red background
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(cloud)
vis.get_render_option().background_color = np.array([1, 1, 1])
vis.run()
vis.destroy_window()

# SHOW INDIVIDUAL FRUMTUMS
for label, frustum_points in max_frustum_points:
    point_cloud = np.asarray(cloud_copy.points)
    filtered_points = filter_points_inside_frustum(point_cloud, frustum_points)
    if filtered_points.size > 0:
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        o3d.visualization.draw_geometries([filtered_pcd])
        save = input("Save? (y/n)")
        if save.lower() == 'y':
            unique_id = uuid.uuid4()
            o3d.io.write_point_cloud(f'./filtered_point_clouds/{unique_id}-{label}.pcd', filtered_pcd)
            print('Saved')

input("Press Enter to continue...")

