import lpips
import numpy as np
import torch
from PIL import Image
import os
import statistics

# Loaded pretrained model
loss_fn = lpips.LPIPS(net='vgg')

# Empty array to store perceptual losses
perceptual_loss_array = []

# Define a list of image file extensions
img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

# Define the paths to the folders containing the images
src_folder_path = "/home/yhsharaf/Desktop/MBZUAI-ML703/BiwiDataset/faces_0/01"
dst_folder_path = "/home/yhsharaf/Desktop/MBZUAI-ML703/BiwiDataset/faces_0/01"
# dst_folder_path = src_folder_path+"/merged"
j=0
# Get a list of all the image filenames in each folder
src_imgs = [f for f in os.listdir(src_folder_path) if os.path.splitext(f)[
    1].lower() in img_extensions]
dst_imgs = [f for f in os.listdir(dst_folder_path) if os.path.splitext(f)[
    1].lower() in img_extensions]

for i in range(len(src_imgs)):
    # Set the path to the folder containing the text files
    src_img = os.path.join(src_folder_path, src_imgs[i])
    dst_img = os.path.join(dst_folder_path, dst_imgs[i])

    # Load image as a PIL Image object and convert it to a PyTorch tensor
    src_img = Image.open(src_img)
    tensor_src_img = torch.tensor(np.array(src_img)).permute(2, 0, 1).float().div(255).mul(2).sub(1)

    # Load image as a PIL Image object and convert it to a PyTorch tensor
    dst_img = Image.open(dst_img)
    tensor_dst_img = torch.tensor(np.array(dst_img)).permute(2, 0, 1).float().div(255).mul(2).sub(1)

    # Calculate the LPIPS distance between the two images
    d = loss_fn.forward(tensor_src_img, tensor_dst_img)
    tensor_value = d[0][0][0][0].item()
    perceptual_loss_array.append(tensor_value)

with open('Perceptual_Loss_results.txt', 'w') as f:
    f.write("Perceptual_Loss_Mean: " +
            str(statistics.mean(perceptual_loss_array)))