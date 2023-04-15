# New
import os
import numpy as np
import shutil


def getClusterSize(value, number_of_clusters, empty_list):

    if(number_of_clusters == 1):
        empty_list = []
        empty_list.append(value)
        return empty_list
    else:
        # empty list
        empty_list = []
        # fill array with values to subtract
        for i in range(0, number_of_clusters):
            empty_list.append(int(value/number_of_clusters)+1)

        # i to save index
        i = 0
        while sum(empty_list) > 32:
            if(i == number_of_clusters):
                i = 0
            empty_list[i] = empty_list[i] - 1
            i += 1
        # print(empty_list)
        return empty_list


# setting a fixed seed for np
np.random.seed(42)

# replace with the path to your folder
path = "./BIWISubset/"
txt_files = [f for f in os.listdir(path) if f.endswith('.txt')]
img_files = [f for f in os.listdir(path) if f.endswith('.png')]

sort_array_txt = []
sort_array_imgs = []

for txt_file in txt_files:
    sort_array_txt.append(txt_file)

for img_file in img_files:
    sort_array_imgs.append(img_file)

sort_array_txt.sort()
sort_array_imgs.sort()

cluster_dict = {}

# split
for i in range(0, len(sort_array_txt)):
    if(sort_array_imgs[i].split("_")[1] == sort_array_txt[i].split("_")[1]):
        cluster_dict[sort_array_imgs[i].split("_")[1]] = [
            path+sort_array_txt[i], path+sort_array_imgs[i], '']

# create a dictionary comprehension with empty lists as default values
n_clusters = [1, 3, 5, 7, 9, 11]
for n_cluster in n_clusters:
    make_key = []

    for i in range(0, n_cluster):
        make_key.append(str(i))

    cluster_key = {key: [] for key in make_key}

    # Set the path to the folder containing the text files
    folder_path = "./BIWISubset/"

    # Empty array
    empty_array = []
    i = 0
    # Loop over each file in the folder
    for key in cluster_dict:
        with open(cluster_dict[key][0], "r") as f:

            # Read the last three lines of the file
            lines = f.readlines()

            for i, line in enumerate(lines):
                if str(n_cluster)+"-KMeans-Cluster:" in line:
                    kmeans_index = i
                    break

            next_line = lines[kmeans_index].strip()
            # get the line after "XYZ:" and remove any leading/trailing whitespace
            next_line = next_line.replace(
                str(n_cluster)+"-KMeans-Cluster: ", "")
            # print(next_line)
            belong_to_cluster = int(next_line)
            # print(f"x: {x}, y: {y}, z: {z}")

            # Store the values in an dictionary and print the array
            cluster_dict[key][2] = belong_to_cluster
            cluster_key[str(belong_to_cluster)].append(str(key))

            i = 1
    set_sample_size = 32
    get_cluster_size = []
    get_cluster_size = getClusterSize(
        set_sample_size, n_cluster, get_cluster_size)

    random_clusters_index = []
    for i in range(0, n_cluster):
        if(int(get_cluster_size[i]) > len(cluster_key[str(i)])):
            random_clusters_index.append(np.random.choice(
                cluster_key[str(i)], size=get_cluster_size[i], replace=True))
        else:
            random_clusters_index.append(np.random.choice(
                cluster_key[str(i)], size=get_cluster_size[i], replace=False))

    for i in range(0, n_cluster):
        for j in range(0, len(random_clusters_index[i])):
            source_location_txt = str(
                cluster_dict[random_clusters_index[i][j]][0])
            source_location_img = str(
                cluster_dict[random_clusters_index[i][j]][1])
        # source_location = '/path/to/source/file.txt'
            destination_location = './Clusters/KMeans/'+str(n_cluster)

            shutil.copy(source_location_txt, destination_location)
            shutil.copy(source_location_img, destination_location)
