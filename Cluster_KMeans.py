# New
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# replace with the path to your folder
path = "/home/yhsharaf/Desktop/MBZUAI-ML703/BIWISubset"
txt_files = [f for f in os.listdir(path) if f.endswith('.txt')]

sort_array = []

for txt_file in txt_files:
    sort_array.append(txt_file)

sort_array.sort()

# Set the path to the folder containing the text files
folder_path = "/home/yhsharaf/Desktop/MBZUAI-ML703/BIWISubset/"

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

# Fit a KMeans model
n_clusters = [1, 2, 4, 6, 8, 10, 12]
for i in range(0, len(n_clusters)):
    model = KMeans(n_clusters[i], random_state=42)
    model.fit(X)

    # # Define colors for each cluster
    # colors = ['gold',
    # 'orange',
    # 'orangered',
    # 'greenyellow',
    # 'turquoise',
    # 'cyan',
    # 'dodgerblue',
    # 'blue',
    # 'blueviolet',
    # 'fuchsia',
    # 'deeppink',
    # 'crimson']

    # # Plot the clusters in 3D with specific colors
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # for i in range(len(X)):
    #     ax.scatter(X[i, 0], X[i, 1], X[i, 2], c=colors[model.labels_[i]], s=50,marker='.')
    # # Set the plot limits and labels
    # ax.set_xlim([0, 1])
    # ax.set_ylim([0, 1])
    # ax.set_zlim([0, 1])
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')
    # # Set the viewing angle
    # ax.view_init(elev=45, azim=180+45)

    # plt.show()

    label = model.predict(normalized_array)  # predicted cluster label
    # print(label)
    # print("New data point belongs to cluster:", label)

    # Save File
    for j in range(len(sort_array)):
        file_path = os.path.join(folder_path, sort_array[j])
        with open(file_path, "a") as f:
            f.write('\n')
            f.write(str(n_clusters[i]) + '-KMeans-Cluster: ' + str(label[j]))
            f.write('\n')
            # print(label[i])
