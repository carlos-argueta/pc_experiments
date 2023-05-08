import numpy as np
import quaternion

from ultralytics import YOLO


#import pypotree

import open3d as o3d

from scipy.spatial import ConvexHull, Delaunay

from copy import deepcopy

import random

def is_point_inside_frustum(point, delaunay):
    return delaunay.find_simplex(point) >= 0

def filter_points_inside_frustum(point_cloud, frustum_points):
    hull = ConvexHull(frustum_points)
    delaunay = Delaunay(frustum_points[hull.vertices])
    return np.array([point for point in point_cloud if is_point_inside_frustum(point, delaunay)])


# Helper functions
def compute_frustum_vertices(intrinsics, bbox, depth_range, rotation, translation):
    # Extract the intrinsic parameters
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Compute the 2D bounding box corners (in image coordinates)
    x1, y1, x2, y2 = bbox

    # Define the 3D bounding box corners in camera coordinates (z = depth_range[0] or z = depth_range[1])
    points_3d_cam = np.array([
        [(x1 - cx) * depth_range[0] / fx, (y1 - cy) * depth_range[0] / fy, depth_range[0]],
        [(x1 - cx) * depth_range[1] / fx, (y1 - cy) * depth_range[1] / fy, depth_range[1]],
        [(x2 - cx) * depth_range[0] / fx, (y1 - cy) * depth_range[0] / fy, depth_range[0]],
        [(x2 - cx) * depth_range[1] / fx, (y1 - cy) * depth_range[1] / fy, depth_range[1]],
        [(x1 - cx) * depth_range[0] / fx, (y2 - cy) * depth_range[0] / fy, depth_range[0]],
        [(x1 - cx) * depth_range[1] / fx, (y2 - cy) * depth_range[1] / fy, depth_range[1]],
        [(x2 - cx) * depth_range[0] / fx, (y2 - cy) * depth_range[0] / fy, depth_range[0]],
        [(x2 - cx) * depth_range[1] / fx, (y2 - cy) * depth_range[1] / fy, depth_range[1]],
    ])

    # Apply the given rotation and translation
    rotation_matrix = quaternion.as_rotation_matrix(rotation)
    points_3d_transformed = (rotation_matrix @ points_3d_cam.T).T + translation

    return points_3d_transformed

def back_project_2D_to_3D(x, y, depth, K, R=None, t=None):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    X_cam = (x - cx) * depth / fx
    Y_cam = (y - cy) * depth / fy
    Z_cam = depth

    P_cam = np.array([X_cam, Y_cam, Z_cam])

    if R is not None and t is not None:
        P_world = np.dot(R.T, (P_cam - t))
        return P_world

    return P_cam

def bbox_2D_to_frustum_simple(bbox_2D, K, z_min, z_max, R=None, t=None):
    x1, y1, x2, y2 = bbox_2D
    
    # Back-project the 2D bounding box corners to 3D points
    P1_zmin = back_project_2D_to_3D(x1, y1, z_min, K, R, t)
    P2_zmin = back_project_2D_to_3D(x2, y1, z_min, K, R, t)
    P3_zmin = back_project_2D_to_3D(x2, y2, z_min, K, R, t)
    P4_zmin = back_project_2D_to_3D(x1, y2, z_min, K, R, t)

    P1_zmax = back_project_2D_to_3D(x1, y1, z_max, K, R, t)
    P2_zmax = back_project_2D_to_3D(x2, y1, z_max, K, R, t)
    P3_zmax = back_project_2D_to_3D(x2, y2, z_max, K, R, t)
    P4_zmax = back_project_2D_to_3D(x1, y2, z_max, K, R, t)

    frustum_points = np.array([P1_zmin, P2_zmin, P3_zmin, P4_zmin, P1_zmax, P2_zmax, P3_zmax, P4_zmax])

    return frustum_points

def bbox_2D_to_frustum(bbox_2D, K, z_min, z_max, R=None, t=None, expand_ratio=0.0):
    x1, y1, x2, y2 = bbox_2D
    
    # Calculate the width and height of the original bounding box
    width = x2 - x1
    height = y2 - y1
    
    # Update the bounding box corners based on the expand_ratio
    x1 = x1 - width * expand_ratio / 2
    x2 = x2 + width * expand_ratio / 2
    y1 = y1 - height * expand_ratio / 2
    y2 = y2 + height * expand_ratio / 2
    
    # Back-project the 2D bounding box corners to 3D points
    P1_zmin = back_project_2D_to_3D(x1, y1, z_min, K, R, t)
    P2_zmin = back_project_2D_to_3D(x2, y1, z_min, K, R, t)
    P3_zmin = back_project_2D_to_3D(x2, y2, z_min, K, R, t)
    P4_zmin = back_project_2D_to_3D(x1, y2, z_min, K, R, t)

    P1_zmax = back_project_2D_to_3D(x1, y1, z_max, K, R, t)
    P2_zmax = back_project_2D_to_3D(x2, y1, z_max, K, R, t)
    P3_zmax = back_project_2D_to_3D(x2, y2, z_max, K, R, t)
    P4_zmax = back_project_2D_to_3D(x1, y2, z_max, K, R, t)

    frustum_points = np.array([P1_zmin, P2_zmin, P3_zmin, P4_zmin, P1_zmax, P2_zmax, P3_zmax, P4_zmax])

    return frustum_points





# Load Yolo V8 model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Inference on an image
date_time_str = "lidar_2022_05_05_16_04_04.9599"
image_src = f"pcds/{date_time_str}.jpg"
cloud_src = f"pcds/{date_time_str}.pcd" #"/home/carlos/Documents/point_cloud_experiments/frustum/pcds/lidar_2022_05_05_15_58_52.5607.pcd"
results = model.predict(source=image_src, show=True) 

# Load and visualize point cloud
# Load the PCD file using Open3D
cloud = o3d.io.read_point_cloud(cloud_src)
cloud_copy = deepcopy(cloud)
#pcd = #cloud.voxel_down_sample(voxel_size=0.05)
print(cloud)
o3d.visualization.draw_geometries([cloud])

# print results
for result in results:
    # Detection
    print(result.boxes.xyxy.numpy())   # box with xyxy format, (N, 4)
    #print(result.boxes.dfdfdfddfxywh.numpy())   # box with xywh format, (N, 4)
    result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
    result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
    print(result.boxes.conf.numpy())   # confidence score, (N, 1)
    print(result.boxes.cls.numpy())    # cls, (N, 1)

# Get the frustum points
K = np.array([[700, 0, 640/2], [0, 700, 384/2], [0, 0, 1]])  # Camera intrinsic matrix (fx, fy, cx, cy)

K = np.array([[261.6734924316406, 0.0, 342.7066345214844], [0.0, 261.6734924316406, 186.6942596435547], [0.0, 0.0, 1.0]])
q = quaternion.from_float_array([0.5, 0.5, -0.5, 0.5])  # Camera rotation matrix 
#q = quaternion.from_float_array([ 0.707106781186548, 0, 0, -0.707106781186547 ])
# Convert the quaternion to a rotation matrix
R = quaternion.as_rotation_matrix(q)
print("R", R)
#R =  np.array([
#    [0, 1, 0],
#    [0, 0, -1],
#    [1, 0, 0]
#])
#R = np.eye(3)
t = np.array([0.087, 0.060, -0.076])                                # Camera translation vector 


print("R", R)


bbox_2Ds = result.boxes.xyxy.numpy()               # Example 2D bounding box (x1, y1, x2, y2)
labels = result.boxes.cls.numpy()                  # Example class label (N, 1)
z_min = 0.5                                              # Minimum depth (meters)
z_max = 50                                           # Maximum depth (meters)

z_max_range = np.arange(z_min, z_max, 0.5)

person_frustum_points = []
car_frustum_points = []
all_frustum_points = []
max_frustum_points = []
bbox_2D = None

frustum_points = None

valid_labels = [0, 2]
for i, bbox_2D in enumerate(bbox_2Ds):
 
    print(labels[i])
   
    for z_max_i in z_max_range:
        frustum_points = bbox_2D_to_frustum(bbox_2D, K, z_min, z_max_i, R, t, expand_ratio=0.5) #compute_frustum_vertices(K, bbox_2D, [z_min, z_max], q, t) #
        #print(frustum_points)
        #frustum_points[:, 1] = -frustum_points[:, 1]
        
        #if labels[i] not in valid_labels:
        #    continue

        if labels[i] == 2:
        #  break

            #car_frustum_points += frustum_points.tolist()
            car_frustum_points.append(frustum_points)
            
        else:
            #person_frustum_points += frustum_points.tolist()
            person_frustum_points.append(frustum_points)

        all_frustum_points.append((labels[i], frustum_points))

        if z_max_i == z_max_range[-1]:
            max_frustum_points.append((labels[i], frustum_points))

car_frustum_points = np.array(car_frustum_points)
person_frustum_points = np.array(person_frustum_points)

# VISUALIZE FRUSTUMS
# Set the color of original points to white (1, 1, 1)
num_original_points = len(cloud.points)
white_colors = np.ones((num_original_points, 3))
cloud.colors = o3d.utility.Vector3dVector(white_colors)

car_frustum_points = np.float64(car_frustum_points)
person_frustum_points = np.float64(person_frustum_points)
#print(all_frustum_points)
#print(frustum_points.shape)
#print(type(frustum_points))
#print(frustum_points.dtype)

def generate_random_color():
    return (random.random(), random.random(), random.random())

num_labels = 80
label_colors = [generate_random_color() for _ in range(num_labels)]

for label, frustum_points in all_frustum_points:
    
    new_points = frustum_points

    color = label_colors[int(label)]

    # Convert new points to an Open3D point cloud
    new_cloud = o3d.geometry.PointCloud()
    new_cloud.points = o3d.utility.Vector3dVector(new_points)

    num_new_points = len(new_points)
    some_colors = np.zeros((num_new_points, 3))
    some_colors[:] = color
    new_cloud.colors = o3d.utility.Vector3dVector(some_colors)

    # Append new points to the existing point cloud
    cloud += new_cloud


# Visualize the point cloud with the original points in white and the new points in black
#o3d.visualization.draw_geometries([cloud])

# Create a custom visualization window with a red background
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(cloud)
vis.get_render_option().background_color = np.array([1, 0, 0])

# Run the visualization loop
vis.run()

# Close the visualization window
vis.destroy_window()

# SHOW INDIVIDUAL FRUMTUMS

for label, frustum_points in max_frustum_points:
    #frustum_points = frustum_points[1]
    #label = frustum_points[0]
    # Convert the Open3D point cloud to a NumPy array
    point_cloud = np.asarray(cloud_copy.points)

    # Assume frustum_points is the output of the bbox_2D_to_frustum function
    #print(max_frustum_points)
    #frustum_points = max_frustum_points[0][1]
    #print(car_frustum_points.shape)
    #print(frustum_points.shape)
    #print(point_cloud.shape)


    # Filter the point cloud using the frustum
    filtered_points = filter_points_inside_frustum(point_cloud, frustum_points)
    print(filtered_points.shape)

    if filtered_points.shape[0] == 0:
        continue

    # Create a new Open3D point cloud with the filtered points
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    print(f'Original point cloud size: {len(point_cloud)}')
    print(f'Filtered point cloud size: {len(filtered_points)}')
    print('label ', label)

    o3d.visualization.draw_geometries([filtered_pcd])


input("Press Enter to continue...")