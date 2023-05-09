from sixdrepnet import SixDRepNet
import cv2
import numpy as np
import os
import statistics

# Create an instance of the SixdRepNet class
model = SixDRepNet()

# Define a list of image file extensions
img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

# auto_array = ['05_To_04_W_To_W_All', '05_To_04_W_To_W_Cluster_1', '05_To_04_W_To_W_GMM_5', '05_To_04_W_To_W_GMM_7', '05_To_04_W_To_W_GMM_9', '05_To_04_W_To_W_GMM_11', '05_To_04_W_To_W_Kmeans_5', '05_To_04_W_To_W_Kmeans_7', '05_To_04_W_To_W_Kmeans_9', '05_To_04_W_To_W_Kmeans_11', '05_To_04_W_To_W_Right_Left', '05_To_04_W_To_W_Static', '05_To_04_W_To_W_Up_Down']
# auto_array = ['05_To_13_W_To_M_All', '05_To_13_W_To_M_Cluster_1', '05_To_13_W_To_M_GMM_5', '05_To_13_W_To_M_GMM_7', '05_To_13_W_To_M_GMM_9', '05_To_13_W_To_M_GMM_11', '05_To_13_W_To_M_Kmeans_5', '05_To_13_W_To_M_Kmeans_7', '05_To_13_W_To_M_Kmeans_9', '05_To_13_W_To_M_Kmeans_11', '05_To_13_W_To_M_Right_Left', '05_To_13_W_To_M_Static', '05_To_13_W_To_M_Up_Down' ]
auto_array = ['05_To_18_SW_To_SW_All', '05_To_18_SW_To_SW_Cluster_1', '05_To_18_SW_To_SW_GMM_5', '05_To_18_SW_To_SW_GMM_7', '05_To_18_SW_To_SW_GMM_9', '05_To_18_SW_To_SW_GMM_11', '05_To_18_SW_To_SW_Kmeans_5',
              '05_To_18_SW_To_SW_Kmeans_7', '05_To_18_SW_To_SW_Kmeans_9', '05_To_18_SW_To_SW_Kmeans_11', '05_To_18_SW_To_SW_Right_Left', '05_To_18_SW_To_SW_Static', '05_To_18_SW_To_SW_Up_Down']
for i in range(0, len(auto_array)):
    # Empty array to store poses score
    poses_score_array = []
    # Before src_folder_path to save text files in
    before_src_path = "/home/youssef.sharaf/Desktop/DeepFaceLab/"+auto_array[i]
    # Current src_folder_path to iterate
    current_src_path = "/home/youssef.sharaf/Desktop/DeepFaceLab/" + \
        auto_array[i]+"/21_Test"
    # Define the paths to the folders containing the images
    src_folder_path = current_src_path+"/face_track_crop"
    # dst_folder_path = "/home/youssef.sharaf/Desktop/DeepFaceLab/05_To_04_W_To_W_All/data_dst/merged"
    dst_folder_path = "/home/youssef.sharaf/Desktop/MBZUAI-ML703/21_GT"
    # Get a list of all the image filenames in each folder and sort them by name
    src_imgs = sorted([f for f in os.listdir(src_folder_path)
                      if os.path.splitext(f)[1].lower() in img_extensions])
    dst_imgs = sorted([f for f in os.listdir(dst_folder_path)
                      if os.path.splitext(f)[1].lower() in img_extensions])

    for j in range(len(src_imgs)):
        # Set the path to the folder containing the text files
        src_img = os.path.join(src_folder_path, src_imgs[j])
        dst_img = os.path.join(dst_folder_path, dst_imgs[j])

        # Load the two input images
        src_img_load = cv2.imread(src_img)
        dst_img_load = cv2.imread(dst_img)

        # Compute the pose between the two images
        pitch1, yaw1, roll1 = model.predict(src_img_load)
        pitch2, yaw2, roll2 = model.predict(dst_img_load)
        # Define two vectors of yaw, roll, and pitch angles
        vec1 = np.array([yaw1, roll1, pitch1])
        vec2 = np.array([yaw2, roll2, pitch2])

        # Calculate the Euclidean distance between the two vectors
        distance = np.linalg.norm(vec1 - vec2)
        poses_score_array.append(distance)

    with open(before_src_path+'/Poses_Score_results.txt', 'w') as f:
        f.write("Poses_Score_Mean: " +
                str(statistics.mean(poses_score_array)))
        print("Poses_Score_Mean: " +
              str(statistics.mean(poses_score_array)))
