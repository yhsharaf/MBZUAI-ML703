import tensorflow as tf
import statistics
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import os

# Empty array to store perceptual losses
perceptual_loss_array = []

# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet', include_top=False)

# Define a list of image file extensions
img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

# Define the paths to the folders containing the images
src_folder_path = "/home/yhsharaf/Desktop/MBZUAI-ML703/BiwiDataset/faces_0/01"
dst_folder_path = "/home/yhsharaf/Desktop/MBZUAI-ML703/BiwiDataset/faces_0/01"
# dst_folder_path = src_folder_path+"/merged"

# Get a list of all the image filenames in each folder
src_imgs = [f for f in os.listdir(src_folder_path) if os.path.splitext(f)[
    1].lower() in img_extensions]
dst_imgs = [f for f in os.listdir(dst_folder_path) if os.path.splitext(f)[
    1].lower() in img_extensions]

for i in range(len(src_imgs)):
    # Set the path to the folder containing the text files
    src_img = os.path.join(src_folder_path, src_imgs[i])
    dst_img = os.path.join(dst_folder_path, dst_imgs[i])

    # Load the images to compare
    src_img_load = image.load_img(src_img, target_size=(224, 224))
    dst_img_load = image.load_img(dst_img, target_size=(224, 224))

    # Convert the images to arrays
    src_array = image.img_to_array(src_img_load)
    dst_array = image.img_to_array(dst_img_load)

    # Preprocess the images for input to the VGG16 model
    src_array = preprocess_input(src_array)
    dst_array = preprocess_input(dst_array)

    # Use the VGG16 model to extract feature maps from the images
    features1 = model.predict(np.array([src_array]))
    features2 = model.predict(np.array([dst_array]))

    # Compute the mean squared error between the feature maps
    mse = np.mean((features1 - features2) ** 2)
    perceptual_loss = mse / 2.0
    perceptual_loss_array.append(perceptual_loss)

with open('Perceptual_Loss_results.txt', 'w') as f:
    f.write("Perceptual_Loss_Mean: " +
            str(statistics.mean(perceptual_loss_array)))
