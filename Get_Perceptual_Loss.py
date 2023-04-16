import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet', include_top=False)

# Load the images to compare
img1 = image.load_img('/home/yhsharaf/Desktop/MBZUAI-ML703/Clusters/KMeans/7/frame_00023_rgb.png', target_size=(224, 224))
img2 = image.load_img('/home/yhsharaf/Desktop/MBZUAI-ML703/Clusters/KMeans/7/frame_00062_rgb.png', target_size=(224, 224))

# Convert the images to arrays
x1 = image.img_to_array(img1)
x2 = image.img_to_array(img2)

# Preprocess the images for input to the VGG16 model
x1 = preprocess_input(x1)
x2 = preprocess_input(x2)

# Use the VGG16 model to extract feature maps from the images
features1 = model.predict(np.array([x1]))
features2 = model.predict(np.array([x2]))

# Compute the mean squared error between the feature maps
mse = np.mean((features1 - features2) ** 2)
perceptual_loss = mse / 2.0

print('The perceptual loss between the two images is:', perceptual_loss)
