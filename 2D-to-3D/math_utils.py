import quaternion
import numpy as np 

# Define the quaternion
q = quaternion.from_float_array([0.5, 0.5, -0.5, 0.5])

# Convert the quaternion to Euler angles (in radians)
euler_angles_rad = quaternion.as_euler_angles(q)

# Convert radians to degrees
euler_angles_deg = np.degrees(euler_angles_rad)

print("Euler angles in radians:", euler_angles_rad)
print("Euler angles in degrees:", euler_angles_deg)



# Define the Euler angles in degrees
euler_angles_deg = np.array([90, -90, -10])

# Convert degrees to radians
euler_angles_rad = np.radians(euler_angles_deg)

# Convert the Euler angles (in radians) to a quaternion
q = quaternion.from_euler_angles(euler_angles_rad)

print("Quaternion:", q)


import numpy as np
from scipy.spatial.transform import Rotation

# Define the rotation matrix
rotation_matrix = np.array([
    [0, 1, 0],
    [0, 0, -1],
    [1, 0, 0]
])

# Convert the rotation matrix to Euler angles
rot = Rotation.from_matrix(rotation_matrix)
euler_angles_rad = rot.as_euler('xyz', degrees=True)

print("Euler angles (in degrees):", euler_angles_rad)


