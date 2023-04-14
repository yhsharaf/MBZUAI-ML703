import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

# Set the path to the folder containing the text files
folder_path = "/home/yhsharaf/Desktop/MBZUAI-ML703/TestingWater"

#Empty array
empty_array = []

# Loop over each file in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a text file
    if filename.endswith(".txt"):
        # Open the file in read mode
        with open(os.path.join(folder_path, filename), "r") as f:
            # Read the last three lines of the file
            lines = f.readlines()[-1:]

            # Extract the x, y, and z values from the lines
            for line in lines:
                values = line.strip().split(",")
                x = float(values[0])
                y = float(values[1])
                z = float(values[2])
                # print(f"x: {x}, y: {y}, z: {z}")

            # Store the values in an array and print the array
            empty_array.append([x,y,z])
print(empty_array)

#Normalize the array
X = np.array(empty_array)
min_val = X.min()
max_val = X.max()
normalized_array = (X - min_val) / (max_val - min_val)
print(normalized_array)

#Make X the normalized array to fit in the data later
X = normalized_array

# Fit a KMeans model
model = KMeans(n_clusters=4, random_state=42)
model.fit(X)

# Plot the clusters in 3D
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=model.labels_, s=100,marker='o', edgecolor='black')
# Set the plot limits and labels
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
# ax.scatter(-10, 0, 10, s=100, c='r', marker='o')
# Set the viewing angle
# ax.view_init(elev=30, azim=45)
ax.view_init(elev=45, azim=180+45)

plt.show()

