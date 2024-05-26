import json
import os.path
import time
import cv2
import numpy as np
import glob
import cv2 as cv
from skimage import segmentation
import random
import pathlib

BASE_DIR = pathlib.Path(__file__).parent


metadata = BASE_DIR.joinpath('CV_DATA').joinpath('SuperviselyExports').joinpath('skincancer_training').joinpath('skincancer_source_images').joinpath('annotations').joinpath('instances.json')
print(metadata)

image_base_path = BASE_DIR.joinpath('CV_DATA').joinpath('SuperviselyExports').joinpath('skincancer_training').joinpath('skincancer_source_images').joinpath('images')

with open(metadata, 'r') as reader:
    instances = json.load(reader)

 

images = instances.get('images')
slim_images = {image.get('id'): image.get('file_name') for image in images}
annotations = instances.get('annotations')

 

i = 0

 

 
j = 0
for id, path in slim_images.items():

 
    image_path = image_base_path.joinpath(path)
    #image_path = f"testset/images/{path}"
    img = cv2.imread(str(image_path))
    

    image_instances = [instance for instance in annotations if instance.get('image_id') == id]

    mask_cum = np.zeros(img.shape, dtype=np.uint8)
                    
    for instance in image_instances:
        # FETCH COCO INFORMATION FROM ANNOTATIONS & LOAD SOURCE IMAGE FOR FOREGROUND EXTRACTION
        segmentation = instance.get('segmentation')
        bbox = instance.get('bbox')

 
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        num_pts = int(len(segmentation[0])/2)
        segmentation = np.reshape(segmentation[0], (num_pts, 2))
        segmentation = np.int32([segmentation])
        mask = cv2.fillPoly(mask, segmentation, (255))

        color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
        mask_color = np.zeros(img.shape, dtype=np.uint8)
        mask_color = cv2.fillPoly(mask_color, segmentation, color)

        mask_cum = cv2.add(mask_cum,mask_color)
 

        res = cv2.bitwise_and(img, img, mask=mask)
        rect = cv2.boundingRect(segmentation)

 

        cropped = res[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

 

        b_channel, g_channel, r_channel = cv2.split(cropped)
        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype)*255

 

        mask_crop = mask[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        alpha_channel[mask_crop[:] == 0] = 0

 

        cropped = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

 

        # filename, ext = os.path.splitext(image_path)

        defect_path = f"defects/corrosion_{i}.png"

        cv2.imwrite(defect_path, cropped)
        i += 1


    
    #mask_path =  BASE_DIR.joinpath('CV_DATA').joinpath('skincancer_source_data').joinpath('trainB').joinpath('{j}.png')
    mask_path = f"CV_DATA/skincancer_source_data/trainB/{j}.png"
    img_path = f"CV_DATA/skincancer_source_data/trainA/{j}.png"
    cv2.imwrite(mask_path,mask_cum)
    cv2.imwrite(img_path,img)
    j += 1 
        #cv2.imwrite('mask',mask)
        
