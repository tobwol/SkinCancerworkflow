import cv2
import io
import os
import time
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import onnxruntime
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def transform_image(image, IMAGE_SIZE):
    transform_pipeline = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ]
    )
    return transform_pipeline(image).unsqueeze(0)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path',dest='dataset_path',
                        help='Parent folder with folders to process (usually source data in trainA (images) & trainB (masks))')
    parser.add_argument('--resize', dest='target_image_size',
                        help='Sets image size to align with MRCNN Model Input')
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    image_size = int(args.target_image_size)

    image_directory = sorted(os.listdir(dataset_path.joinpath('trainA')))
    mask_directory = sorted(os.listdir(dataset_path.joinpath('trainB')))

    for image, mask in zip(image_directory, mask_directory):

        image_name = dataset_path.joinpath('trainA').joinpath(str(image))
        mask_name = dataset_path.joinpath('trainB').joinpath(str(mask))

        with open(image_name, 'rb') as reader:
            file_bytes = reader.read()

        img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        mask = Image.open(mask_name).convert("RGBA")

        orig_image_size = img.size

        final_image = img.resize((image_size, image_size))
        mask_found = mask.resize((image_size, image_size))

        final_image.save(image_name)
        mask_found.save(mask_name)
        final_image.close()
        mask_found.close()
