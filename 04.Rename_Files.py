import os

folder_path = '/home/yhsharaf/Documents/DeepFaceLab/DeepFaceLab_Linux/workspace/data_src'
count = 0
for filename in os.listdir(folder_path):
    if filename.endswith('.png'):
        new_filename = '{:05d}.png'.format(count)
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
        count += 1
