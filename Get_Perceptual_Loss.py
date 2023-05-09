import lpips
import numpy as np
import torch
from PIL import Image
import os
import statistics

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loaded pretrained model
loss_fn = lpips.LPIPS(net='vgg').to(device)

# Define a list of image file extensions
img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']


# auto_array = ['05_To_04_W_To_W_All', '05_To_04_W_To_W_Cluster_1', '05_To_04_W_To_W_GMM_5', '05_To_04_W_To_W_GMM_7', '05_To_04_W_To_W_GMM_9', '05_To_04_W_To_W_GMM_11', '05_To_04_W_To_W_Kmeans_5', '05_To_04_W_To_W_Kmeans_7', '05_To_04_W_To_W_Kmeans_9', '05_To_04_W_To_W_Kmeans_11', '05_To_04_W_To_W_Right_Left', '05_To_04_W_To_W_Static', '05_To_04_W_To_W_Up_Down']
# auto_array = ['05_To_13_W_To_M_All', '05_To_13_W_To_M_Cluster_1', '05_To_13_W_To_M_GMM_5', '05_To_13_W_To_M_GMM_7', '05_To_13_W_To_M_GMM_9', '05_To_13_W_To_M_GMM_11', '05_To_13_W_To_M_Kmeans_5', '05_To_13_W_To_M_Kmeans_7', '05_To_13_W_To_M_Kmeans_9', '05_To_13_W_To_M_Kmeans_11', '05_To_13_W_To_M_Right_Left', '05_To_13_W_To_M_Static', '05_To_13_W_To_M_Up_Down' ]
auto_array = ['05_To_18_SW_To_SW_All', '05_To_18_SW_To_SW_Cluster_1', '05_To_18_SW_To_SW_GMM_5', '05_To_18_SW_To_SW_GMM_7', '05_To_18_SW_To_SW_GMM_9', '05_To_18_SW_To_SW_GMM_11', '05_To_18_SW_To_SW_Kmeans_5',
              '05_To_18_SW_To_SW_Kmeans_7', '05_To_18_SW_To_SW_Kmeans_9', '05_To_18_SW_To_SW_Kmeans_11', '05_To_18_SW_To_SW_Right_Left', '05_To_18_SW_To_SW_Static', '05_To_18_SW_To_SW_Up_Down']
for i in range(0, len(auto_array)):
    # Empty array to store perceptual losses
    perceptual_loss_array = []
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

        # Load image as a PIL Image object and convert it to a PyTorch tensor
        src_img = Image.open(src_img)
        tensor_src_img = torch.tensor(np.array(src_img)).permute(
            2, 0, 1).float().div(255).mul(2).sub(1).to(device)

        # Load image as a PIL Image object and convert it to a PyTorch tensor
        dst_img = Image.open(dst_img)
        tensor_dst_img = torch.tensor(np.array(dst_img)).permute(
            2, 0, 1).float().div(255).mul(2).sub(1).to(device)
        # Calculate the LPIPS distance between the two images
        d = loss_fn.forward(tensor_src_img, tensor_dst_img)
        tensor_value = d[0][0][0][0].item()
        perceptual_loss_array.append(tensor_value)

    with open(before_src_path+'/Perceptual_Loss_results_vgg.txt', 'w') as f:
        f.write("Perceptual_Loss_Mean: " +
                str(statistics.mean(perceptual_loss_array)))
        print("Perceptual_Loss_Mean: " +
              str(statistics.mean(perceptual_loss_array)))
