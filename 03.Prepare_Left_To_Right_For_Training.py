import os
import glob
import re
import shutil

folder_path = '/home/yhsharaf/Desktop/MBZUAI-ML703/BiwiDataset/faces_0/01'  # replace with the path to your folder
xyz_pattern = r"XYZ:\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)"
xyz_dict = {}

# create the output folder if it does not exist
output_folder_path = '/home/yhsharaf/Desktop/MBZUAI-ML703/Left_Right'  # replace with the path to your output folder
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

for file_path in glob.glob(os.path.join(folder_path, '*.txt')):
    with open(file_path, 'r') as file:
        content = file.read()
        match = re.search(xyz_pattern, content)
        if match:
            x, y, z = map(float, match.groups())
            if -8 <= x <= 8 and -90 <= y <= 90 and -8 <= z <= 8:
                output_file_path = os.path.join(output_folder_path, os.path.basename(file_path).replace("pose.txt","rgb.png"))
                shutil.copy2(file_path.replace("pose.txt","rgb.png"), output_file_path)
                xyz_dict[os.path.basename(output_file_path)] = {'x': x, 'y': y, 'z': z}
        else:
            print(f"No XYZ values found in {os.path.basename(file_path)}")

print(xyz_dict)
