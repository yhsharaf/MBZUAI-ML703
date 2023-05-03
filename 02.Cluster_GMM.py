# New
import os
import numpy as np
from sklearn.mixture import GaussianMixture

# Go throught all the files in the directory of BIWI dataset
main_folder_path = "./BiwiDataset/faces_0/"

for subdir, dirs, files in os.walk(main_folder_path):
    for dir in dirs:
        # Set the path to the folder containing the text files
        folder_path = os.path.join(subdir, dir)+'/'

        txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

        sort_array = []

        for txt_file in txt_files:
            sort_array.append(txt_file)

        sort_array.sort()

        # Empty array
        empty_array = []
        i = 0
        # Loop over each file in the folder
        for i in range(len(sort_array)):
            file_path = os.path.join(folder_path, sort_array[i])
            with open(file_path, "r") as f:

                # Read the last three lines of the file
                lines = f.readlines()

                for i, line in enumerate(lines):
                    if "XYZ:" in line:
                        xyz_index = i
                        break

                next_line = lines[xyz_index].strip()
                # get the line after "XYZ:" and remove any leading/trailing whitespace
                next_line = next_line.replace("XYZ: ", "").split(",")
                # print(next_line)
                x = float(next_line[0])
                y = float(next_line[1])
                z = float(next_line[2])
                # print(f"x: {x}, y: {y}, z: {z}")

                # Store the values in an array and print the array
                empty_array.append([x, y, z])
                i = 1
        # print(empty_array)

        # Normalize The Values
        X = np.array(empty_array)
        min_val = X.min()
        max_val = X.max()
        normalized_array = (X - min_val) / (max_val - min_val)
        # print(normalized_array)

        X = normalized_array

        # Fit a GMM model
        n_clusters = [1, 3, 5, 7, 9, 11]
        for i in range(0, len(n_clusters)):
            model = GaussianMixture(n_clusters[i], random_state=42)
            model.fit(X)

            label = model.predict(normalized_array)  # predicted cluster label
            # print(label)
            # print("New data point belongs to cluster:", label)

            # Save File
            for j in range(len(sort_array)):
                file_path = os.path.join(folder_path, sort_array[j])
                with open(file_path, "a") as f:
                    f.write('\n')
                    f.write(str(n_clusters[i]) +
                            '-GMM-Cluster: ' + str(label[j]))
                    f.write('\n')
                    # print(label[i])
