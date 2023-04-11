import numpy as np

#Clockwise

def euler_angles_from_rotation_matrix_rr(R):
    # Extract yaw (z-axis rotation), pitch (y-axis rotation), and roll (x-axis rotation) from a 3x3 rotation matrix R
    # Inputs:
    # R: 3x3 rotation matrix
    # Outputs:
    # yaw: rotation angle around z-axis (in radians)
    # pitch: rotation angle around y-axis (in radians)
    # roll: rotation angle around x-axis (in radians)

    if R[0, 2] > 0.998:  # singularity at north pole
        yaw = np.arctan2(R[1, 0], R[1, 1])
        pitch = np.pi/2
        roll = 0
    elif R[0, 2] < -0.998:  # singularity at south pole
        yaw = np.arctan2(R[1, 0], R[1, 1])
        pitch = -np.pi/2
        roll = 0
    else:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    return yaw, pitch, roll 

R_down = np.array([[0.999339 ,0.0362923, -0.00211611 ],
[-0.0254109, 0.655711, -0.754584] ,
[-0.025998, 0.754139, 0.6562 ]])

R_right = np.array([[0.429587, -0.16792, -0.887276 ],
[-0.0321034, 0.979098, -0.200841], 
[0.902455, 0.114763, 0.415216]])

R_up = np.array([[0.99733, -0.0212466 ,0.069873 ],
[-0.0154851,0.873472, 0.486628] ,
[-0.0713713, -0.48641, 0.870811] ])

R_g = np.array([[0.992404, -0.0457364, 0.114207 ],
[-0.0152238, 0.875537, 0.482912] ,
[-0.122079, -0.480982, 0.868189 ]])

R_x = np.array([[1,0,0]
                 ,[0,0.6,-0.6]
                 ,[0,0.6,0.6]])

R_y = np.array([[1,0,0]
                 ,[0,1,0]
                 ,[0,0,1]])

R_z = np.array([[1,0,0]
                 ,[0,1,0]
                 ,[0,0,1]])

R = R_x @ R_y @ R_z

R_g = np.array([[0.997874, 0.0557298, 0.0338006 ],
[-0.0304369 ,0.856994, -0.514426 ],
[-0.0576358, 0.512304 ,0.856868 ]])

psi, theta, phi = euler_angles_from_rotation_matrix_rr(R_g)

# print("Yaw angle (psi):  radians" , psi)
# print("Pitch angle (theta):  radians" , theta)
# print("Roll angle (phi):  radians" , phi)

print("Roll_z angle (psi):  radians" , psi*(180/np.pi))
print("Yaw_y angle (theta):  radians" , theta*(180/np.pi))
print("Pitch_x angle (phi):  radians" , phi*(180/np.pi))


