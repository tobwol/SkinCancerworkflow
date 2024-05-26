import argparse
import json
import shutil
import warnings
import random
from math import sqrt
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageEnhance
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
import cv2
import os
counter = 1
if __name__ == '__main__':    
    parser = argparse.ArgumentParser()

    parser.add_argument('--bruteforce_source_data', dest='bruteforce_data', action='store',
                        help='Path to the folder containing subfolders backgrounds/foregrounds')
    
    parser.add_argument('--gan_source_data', dest='gan_data', action='store',
                        help='Path to the folder containing subfolders backgrounds/foregrounds')
    
    parser.add_argument('--real_source_data', dest='real_data', action='store',
                        help='Path to the folder containing subfolders backgrounds/foregrounds')  
     
    parser.add_argument('--dataset_size', dest='dataset_size', action='store',
                        help='Size of COCO synthetic dataset')
    
    parser.add_argument('--percentage_bruteforce', dest='percentage_bruteforce', action='store',
                        help='Size of COCO synthetic dataset')
    
    parser.add_argument('--MASKRCNN_training_data', dest='MASKRCNN_training_data', action='store',
                        help='Size of COCO synthetic dataset')
    
    parser.add_argument('--percentage_gan', dest='percentage_gan', action='store',
                        help='Size of COCO synthetic dataset')
    
    parser.add_argument('--percentage_real', dest='percentage_real', action='store',
                        help='Size of COCO synthetic dataset')    
                    
    parser.add_argument('--image_size', dest='image_size', action='store',
                        help='Size of the synthetic images as a square')
    

    arguments = parser.parse_args()

    destination_folder = arguments.MASKRCNN_training_data

    total_images = int(arguments.dataset_size)

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    if not os.path.exists(os.path.join(destination_folder,'images')):
        os.makedirs(os.path.join(destination_folder,'images'))

    if not os.path.exists(os.path.join(destination_folder,'masks')):
        os.makedirs(os.path.join(destination_folder,'masks'))
    
    ### CONVERT TO INPUTS FOR WORKFLOW
    dataset_size = int(arguments.dataset_size)

    source_folders = {arguments.gan_data: arguments.percentage_gan, arguments.real_data: arguments.percentage_real, arguments.bruteforce_data: arguments.percentage_bruteforce}
    
for folder, percentage in source_folders.items():
    num_images_to_copy = int(total_images * float(percentage))  # Convert percentage to float

    if "source_data" in str(folder):
        img_folder_path = os.path.join(folder, 'trainA')
        mask_folder_path = os.path.join(folder, 'trainB')
        num_images_to_copy = len(os.listdir(img_folder_path))  # Convert percentage to float
    else:
        img_folder_path = os.path.join(folder, 'images')
        mask_folder_path = os.path.join(folder, 'masks')
        num_images_to_copy = int(total_images * float(percentage))  # Convert percentage to float
    folder_base_name = os.path.basename(folder)
    # Check if the directories exist
    if not os.path.exists(img_folder_path):
        print(f"Error: Image folder '{img_folder_path}' does not exist.")
        continue

    if not os.path.exists(mask_folder_path):
        print(f"Error: Mask folder '{mask_folder_path}' does not exist.")
        continue

    img_files = os.listdir(img_folder_path)
    mask_files = os.listdir(mask_folder_path)

    # Choose a random sample of files from the list
    random_img_files = random.sample(img_files, min(num_images_to_copy, len(img_files)))

    # Copy the selected random files to the destination folder
    for img_name in random_img_files:
        if "_gan_data" in str(folder):
            common_prefix = img_name.split('_fake.png')[0]
            mask_name = common_prefix + '_real.png'
        else:
            mask_name = img_name
            # Construct the new image and mask names using the counter
        new_img_name = f"{counter}.png"
        new_mask_name = f"{counter}.png"  # Assuming the image and mask have the same name
        new_img_name = f"{counter}.png"
        new_mask_name = f"{counter}.png"        
        counter += 1
        img_source_path = os.path.join(img_folder_path, img_name)
        mask_source_path = os.path.join(mask_folder_path, mask_name)
        
        img_destination_path = os.path.join(destination_folder, 'images', new_img_name)
        mask_destination_path = os.path.join(destination_folder, 'masks', new_mask_name)

        # Debugging output
        print("Copying:", img_source_path, "to", img_destination_path)
        print("Copying:", mask_source_path, "to", mask_destination_path)

        # Copy files
        shutil.copyfile(img_source_path, img_destination_path)
        shutil.copyfile(mask_source_path, mask_destination_path)