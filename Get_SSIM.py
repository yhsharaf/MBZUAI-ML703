from skimage.metrics import structural_similarity as ssim
import cv2
import os
import statistics
import numpy as np
from skimage import io

# Define a list of image file extensions
img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

# auto_array = ['05_To_04_W_To_W_All', '05_To_04_W_To_W_Cluster_1', '05_To_04_W_To_W_GMM_5', '05_To_04_W_To_W_GMM_7', '05_To_04_W_To_W_GMM_9', '05_To_04_W_To_W_GMM_11', '05_To_04_W_To_W_Kmeans_5', '05_To_04_W_To_W_Kmeans_7', '05_To_04_W_To_W_Kmeans_9', '05_To_04_W_To_W_Kmeans_11', '05_To_04_W_To_W_Right_Left', '05_To_04_W_To_W_Static', '05_To_04_W_To_W_Up_Down']
# auto_array = ['05_To_13_W_To_M_All', '05_To_13_W_To_M_Cluster_1', '05_To_13_W_To_M_GMM_5', '05_To_13_W_To_M_GMM_7', '05_To_13_W_To_M_GMM_9', '05_To_13_W_To_M_GMM_11', '05_To_13_W_To_M_Kmeans_5', '05_To_13_W_To_M_Kmeans_7', '05_To_13_W_To_M_Kmeans_9', '05_To_13_W_To_M_Kmeans_11', '05_To_13_W_To_M_Right_Left', '05_To_13_W_To_M_Static', '05_To_13_W_To_M_Up_Down' ]
auto_array = ['05_To_18_SW_To_SW_All', '05_To_18_SW_To_SW_Cluster_1', '05_To_18_SW_To_SW_GMM_5', '05_To_18_SW_To_SW_GMM_7', '05_To_18_SW_To_SW_GMM_9', '05_To_18_SW_To_SW_GMM_11', '05_To_18_SW_To_SW_Kmeans_5',
              '05_To_18_SW_To_SW_Kmeans_7', '05_To_18_SW_To_SW_Kmeans_9', '05_To_18_SW_To_SW_Kmeans_11', '05_To_18_SW_To_SW_Right_Left', '05_To_18_SW_To_SW_Static', '05_To_18_SW_To_SW_Up_Down']
for i in range(0, len(auto_array)):
    # Empty array to store ssim score
    ssim_score_array = []
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

        # Calculate SSIM
        ssim_score = ssim(src_img_load, dst_img_load,
                          multichannel=True, channel_axis=-1)
        # print(ssim_score)
        ssim_score_array.append(ssim_score)

    ssim_score_array_np = np.array(ssim_score_array)

    with open(before_src_path+'/SSIM_Score_results.txt', 'w') as f:
        ssim_score_array_np = np.array(ssim_score_array)
        f.write("SSIM_Score_Mean: " +
                str(np.mean(ssim_score_array_np)))
        print("SSIM_Score_Mean: " +
              str(np.mean(ssim_score_array_np)))
        f.write("\nSSIM_Score_Std: " +
                str(np.std(ssim_score_array_np)))
        print("SSIM_Score_Std: " +
              str(np.std(ssim_score_array_np)))
