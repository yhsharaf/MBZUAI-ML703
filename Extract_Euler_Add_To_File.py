import numpy as np
import os
import glob

# Set the directory containing the text files
dir_path = '/home/yhsharaf/Desktop/ML703/TestingWater'

# Get a list of all the text files in the directory
txt_files = glob.glob(os.path.join(dir_path, '*.txt'))

# Loop over each text file
for file_path in txt_files:
    with open(file_path, 'r') as file:
        # Read the contents of the file
        # Read the first three lines of the file
        first_three_lines = [next(file) for x in range(3)]
        # print(first_three_lines)
        # Split each line of text into a list of strings and convert to floats
    matrix = []
    for line in first_three_lines:
        row = [float(x) for x in line.split()]
        matrix.append(row)
    
    # Convert the list of lists to a numpy array and print it
    matrix = np.array(matrix)
    # print(matrix)
    # Calculate pitch (rotation around X-axis)
    pitch = np.arctan2(-matrix[2,1], np.sqrt(matrix[0,1]**2 + matrix[1,1]**2))

    # Calculate roll (rotation around Y-axis)
    roll = np.arctan2(matrix[2,0], matrix[2,2])

    # Calculate yaw (rotation around Z-axis)
    yaw = np.arctan2(-matrix[1,0], matrix[0,0])

    # Print the results in radians
    # print("Pitch: radians",np.radians(pitch))
    # print("Roll:  radians",np.radians(roll))
    # print("Yaw:   radians",np.radians(yaw))

    pitch = np.radians(pitch)
    roll = np.radians(roll)
    yaw = np.radians(yaw)

    pitch_roll_yaw = pitch,roll,yaw
    # print(pitch_roll_yaw)

    # Append the matrix to the end of the file
    with open(file_path, 'a') as file:
        file.write('\n')
        np.savetxt(file, pitch_roll_yaw)

