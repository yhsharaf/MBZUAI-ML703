import numpy as np
import os
import glob

# Go throught all the files in the directory of BIWI dataset
main_folder_path = "./BiwiDataset/faces_0/"

for subdir, dirs, files in os.walk(main_folder_path):
    for dir in dirs:
        # Set the path to the folder containing the text files
        folder_path = os.path.join(subdir, dir)+'/'

        # Get a list of all the text files in the directory
        txt_files = glob.glob(os.path.join(folder_path, '*.txt'))

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
            pitch = np.arctan2(-matrix[2, 1],
                            np.sqrt(matrix[0, 1]**2 + matrix[1, 1]**2))

            # Calculate roll (rotation around Y-axis)
            roll = np.arctan2(matrix[2, 0], matrix[2, 2])

            # Calculate yaw (rotation around Z-axis)
            yaw = np.arctan2(-matrix[1, 0], matrix[0, 0])

            # Print the results in radians
            # print("Pitch: radians",np.radians(pitch))
            # print("Roll:  radians",np.radians(roll))
            # print("Yaw:   radians",np.radians(yaw))

            pitch = np.degrees(pitch)
            roll = np.degrees(roll)
            yaw = np.degrees(yaw)

            pitch_roll_yaw = str(pitch) + ',' + str(roll) + ',' + str(yaw)

            # print(pitch_roll_yaw)
            # Append the matrix to the end of the file
            with open(file_path, 'a') as file:
                file.write('XYZ: '+pitch_roll_yaw)
                file.write('\n')
