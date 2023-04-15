import numpy as np

# Define the rotation matrix
R = np.array([[0.997874, 0.0557298, 0.0338006],
              [-0.0304369, 0.856994, -0.514426],
              [-0.0576358, 0.512304, 0.856868]])

# Calculate pitch (rotation around X-axis)
pitch = np.arctan2(-R[2, 1], np.sqrt(R[0, 1]**2 + R[1, 1]**2))

# Calculate roll (rotation around Y-axis)
roll = np.arctan2(R[2, 0], R[2, 2])

# Calculate yaw (rotation around Z-axis)
yaw = np.arctan2(-R[1, 0], R[0, 0])

# Print the results in degrees
print("Pitch:  deg", np.degrees(pitch))
print("Roll:  deg", np.degrees(roll))
print("Yaw:  deg", np.degrees(yaw))
