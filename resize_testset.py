from PIL import Image
import os

# Base directory containing images and masks
base_dir = 'test'

# Target size for resizing
target_size = (512, 512)

# Directories containing images and masks
image_dir = os.path.join(base_dir, 'images')
mask_dir = os.path.join(base_dir, 'masks')

# Function to resize images in a directory
def resize_images_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust file formats as needed
            # Open the image file
            image_path = os.path.join(directory, filename)
            img = Image.open(image_path)
            
            # Resize the image
            img_resized = img.resize(target_size, Image.ANTIALIAS)
            
            # Save the resized image, overwriting the original file
            img_resized.save(image_path)

            print(f"Resized {filename} to {target_size}")

# Resize images in the "images" folder
resize_images_in_directory(image_dir)

# Resize images in the "masks" folder
resize_images_in_directory(mask_dir)