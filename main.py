from create_Brute_force import ImageComposition
import os
import pathlib
import shutil
import time
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import subprocess


def convert_arguments(argument_list):
    return [str(arg) for arg in argument_list]
new_brute_force = ImageComposition(100)

BASE_DIR = pathlib.Path(__file__).parent
CUR_DIR = pathlib.Path(__file__).parent

## SETUP MODEL & DATA DIRECTORY OUTSIDE REPO FOLDER IN SAME PARENT
MODEL_DIR = BASE_DIR.joinpath('CV_MODELS')
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = BASE_DIR.joinpath('CV_DATA')
DATA_DIR.mkdir(parents=True, exist_ok=True)

Application_Name = 'skincancer'           # provide a unique application name to generate all folders for workflow
IMAGE_SIZE = 256                    # set this to whatever image size both the gan and mrcnn will use
DATASET_SIZE = 14000
PERCENTAGE_BRUTEFORVCE = 100
PERCENTAGE_GAN = 0
#GAN_SIZE = 0.5
#BRUTEFORCE_SIZE = 0.5
NUMBER_OF_EPOCHS_MRCNN = 20
PREPROCESS_SOURCE_IMAGES = False
BUILD_GAN_IMAGES = False            # If GAN aligned images haven't been generated use this flag
TRAIN_GAN = False
COCO_GENERATE = False
GAN_GENERATE = False
PREPARE_SYNTHETIC_DATA = False
GENERATE_DATASET = False
TRAIN_MRCNN = False
TEST_MRCNN = True
SHOW_MRCNN = False
CONVERT_ONNX = False

GAN_CHECKPOINTS = MODEL_DIR.joinpath(f'{Application_Name}').joinpath('pytorch').joinpath('gan')
MRCNN_CHECKPOINTS = MODEL_DIR.joinpath(f'{Application_Name}').joinpath('pytorch').joinpath('mrcnn')
ONNX_CHECKPOINTS = MODEL_DIR.joinpath(f'{Application_Name}').joinpath('onnx')

labeled_source_data = DATA_DIR.joinpath(f'{Application_Name}_source_data')
coco_source_data = DATA_DIR.joinpath(f'{Application_Name}_coco_data')
gan_synthetic_data = DATA_DIR.joinpath(f'{Application_Name}_gan_data')

bruteforce_source_data = coco_source_data.joinpath('coco_brute_force').joinpath('train')
gan_source_data = gan_synthetic_data.joinpath('train')
real_source_data = labeled_source_data
MASKRCNN_training_data = DATA_DIR.joinpath('MASKRCNN_training_data')


percentage_real = 50



labeled_source_data.mkdir(parents=True, exist_ok=True)
coco_source_data.mkdir(parents=True, exist_ok=True)
gan_synthetic_data.mkdir(parents=True, exist_ok=True)
GAN_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
MRCNN_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
ONNX_CHECKPOINTS.mkdir(parents=True, exist_ok=True)

if PREPROCESS_SOURCE_IMAGES:
    dataset_preprocess_arguments = [
        '--dataset-path', labeled_source_data,
        '--resize', IMAGE_SIZE,
    ]

    try:
        dataset_preprocess_arguments = convert_arguments(dataset_preprocess_arguments)
        dataset_process = subprocess.run(["python3", "preprocess_source_images_for_gan.py"]+dataset_preprocess_arguments,
                                        check=True, capture_output=False, text=True)
        print(dataset_process)
    except subprocess.CalledProcessError as e:
        print(e)

if BUILD_GAN_IMAGES:

    dataset_alignment_arguments = [
        '--dataset-path', labeled_source_data,
        '--resize', IMAGE_SIZE
    ]

    try:
        dataset_alignment_arguments = convert_arguments(dataset_alignment_arguments)
        gan_align_data = subprocess.run(["python3", "make_dataset_aligned.py"]+dataset_alignment_arguments,
                                        check=True, capture_output=False, text=True)
        print(gan_align_data)
    except subprocess.CalledProcessError as e:
        print(e)


if TRAIN_GAN:
    ### Initialize GAN model and Train on Labeled Source Data
    gan_training_arguments = [
        '--dataroot', labeled_source_data,
        '--checkpoints_dir', GAN_CHECKPOINTS,
        '--name', Application_Name+'_pix2pix',
        '--model', 'pix2pix',
        '--direction', 'BtoA',
        '--n_epochs', 40,
        '--n_epochs_decay', 2,
        '--gpu_ids', '0',
        '--preprocess', False,
    ]

    try:
        print("Train GAN")
        gan_training_arguments = convert_arguments(gan_training_arguments)
        gan_training = subprocess.run(["python3", "train_gan.py"]+gan_training_arguments,
                                  check=True, capture_output=False, text=True)
        print(gan_training)
    except subprocess.CalledProcessError as e:
        print(e)

if COCO_GENERATE:
    ### Generate COCO dataset of  brute force generated masks --> INPUT TO Trained GAN For synthetic dataset
    coco_generation_arguments = [
        '--source_data', coco_source_data,
        '--dataset_size', DATASET_SIZE,
        '--image_size', IMAGE_SIZE,
        '--split_ratio_train', .8
    ]
    coco_generation_arguments = convert_arguments(coco_generation_arguments)
    rough_gen = subprocess.run(["python3", "create_synthetic_coco_masks.py"]+coco_generation_arguments,
                               check=True, capture_output=False, text=True)
    print(rough_gen)

coco_masks_path_train = coco_source_data.joinpath('coco_brute_force').joinpath('train').joinpath('masks')
coco_masks_path_val = coco_source_data.joinpath('coco_brute_force').joinpath('val').joinpath('masks')
gan_result_train = gan_synthetic_data.joinpath('coco_results_train')
gan_result_val = gan_synthetic_data.joinpath('coco_results_val')

coco_set = [coco_masks_path_train, coco_masks_path_val]
gan_set = [gan_result_train, gan_result_val]

if GAN_GENERATE:
    start = time.time()
    coco_gan_generation_arguments = [
        '--dataroot', coco_set[0],
        '--checkpoints_dir', GAN_CHECKPOINTS,
        #'--num_test', len(os.listdir(coco_set[0])),
        '--num_test', DATASET_SIZE,
        '--name', Application_Name+'_pix2pix',
        '--model', 'test',
        '--netG', 'unet_256',
        '--preprocess', False,
        '--direction', 'AtoB',
        '--norm', 'batch',
        '--gpu_ids', '0',
        '--results_dir', gan_set[0]
    ]

    coco_gan_generation_arguments = convert_arguments(coco_gan_generation_arguments)
    gan_train_generator = subprocess.run(["python3", "test_gan.py"]+coco_gan_generation_arguments,
                                         check=True, capture_output=False, text=True)
    print(gan_train_generator)
    print(f"Generation of {coco_set[0]} ran for: {time.time()-start}")

    start = time.time()
    coco_gan_generation_arguments = [
        '--dataroot', coco_set[1],
        '--num_test', len(os.listdir(coco_set[1])),
        '--checkpoints_dir', GAN_CHECKPOINTS,
        '--name', Application_Name+'_pix2pix',
        '--model', 'test',
        '--netG', 'unet_256',
        '--preprocess', False,
        '--direction', 'AtoB',
        '--norm', 'batch',
        '--gpu_ids', '0',
        '--results_dir', gan_set[1]
    ]

    coco_gan_generation_arguments = convert_arguments(coco_gan_generation_arguments)
    gan_val_generator = subprocess.run(["python3", "test_gan.py"]+coco_gan_generation_arguments,
                                       check=True, capture_output=False, text=True)
    print(gan_val_generator)
    print(f"Generation of {coco_set[1]} ran for: {time.time()-start}")


if PREPARE_SYNTHETIC_DATA:
    for split in gan_set:
        gan_synthetic_data_input_location = split.joinpath(f'{Application_Name}_pix2pix').joinpath('test_latest').joinpath('images')

        name = 'train' if 'train' in str(split) else 'val'
        split_size = len(os.listdir(gan_synthetic_data_input_location))/2
        gan_set_output_location = gan_synthetic_data.joinpath(f'{name}')
        print(f'Preparing dataset: {name}_{split_size} at {gan_set_output_location}')
        gan_synthetic_data_image_location = gan_set_output_location.joinpath('images')
        gan_synthetic_data_mask_location = gan_set_output_location.joinpath('masks')

        try:
            gan_synthetic_data_image_location.mkdir(parents=True, exist_ok=False)
            gan_synthetic_data_mask_location.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            shutil.rmtree(gan_synthetic_data_image_location)
            shutil.rmtree(gan_synthetic_data_mask_location)
            gan_synthetic_data_image_location.mkdir(parents=True, exist_ok=False)
            gan_synthetic_data_mask_location.mkdir(parents=True, exist_ok=False)

        ### RELOCATE results/application_pix2pix/test_latest/images/
        # into a chip_gan_data folder  -- for both train & validation
        # real (masks) --> train/masks
        # fake (images) --> train/images
        files = sorted(os.listdir(gan_synthetic_data_input_location))
        images = [str(img) for img in files if 'fake' in str(img)]
        masks = [str(mask) for mask in files if 'real' in str(mask)]

        for image, mask in tqdm(zip(images, masks)):
            ### INSERT COLOR CONVERSION HERE
            # Move image to images folder
            shutil.move(gan_synthetic_data_input_location.joinpath(image), gan_synthetic_data_image_location.joinpath(image))
            # Move mask to masks folder
            shutil.move(gan_synthetic_data_input_location.joinpath(mask), gan_synthetic_data_mask_location.joinpath(mask))

if GENERATE_DATASET:
    ### Generate COCO dataset of  brute force generated masks --> INPUT TO Trained GAN For synthetic dataset
    coco_generation_arguments = [
        '--bruteforce_source_data', bruteforce_source_data,
        '--gan_source_data', gan_source_data,
        '--real_source_data', real_source_data,
        '--MASKRCNN_training_data', MASKRCNN_training_data,
        '--percentage_bruteforce', PERCENTAGE_BRUTEFORVCE,
        '--percentage_gan', PERCENTAGE_GAN,
        '--percentage_real', percentage_real,
        '--dataset_size', 800
    ]

    coco_generation_arguments = convert_arguments(coco_generation_arguments)
    rough_gen = subprocess.run(["python3", "shuffel.py"]+coco_generation_arguments,
                               check=True, capture_output=False, text=True)
    print(rough_gen)


if TRAIN_MRCNN:
    train_dir = gan_synthetic_data
    ### Initialize PyTorch MRCNN model for training and Train model
    # # ADD IN SAVING ADJUSTMENT FOR INTERMEDIATE SAVES
    maskrcnn_training_arguments = [
        '--train_dir', gan_synthetic_data.joinpath('train'),
        '--mrcnn_path', MRCNN_CHECKPOINTS,
        '--val_dir', gan_synthetic_data.joinpath('val'),
        '--n_epochs', NUMBER_OF_EPOCHS_MRCNN,
        '--name', Application_Name
    ]

    maskrcnn_training_arguments = convert_arguments(maskrcnn_training_arguments)
    maskrcnn_training = subprocess.run(["python3", "maskrcnn_training.py"]+maskrcnn_training_arguments,
                                         check=True, capture_output=False, text=True)
    print(maskrcnn_training)
print("Checkpoints")
print(MRCNN_CHECKPOINTS)    
if TEST_MRCNN:
    train_dir = gan_synthetic_data
    ### Initialize PyTorch MRCNN model for training and Train model
    # # ADD IN SAVING ADJUSTMENT FOR INTERMEDIATE SAVES
    maskrcnn_test_arguments = [
        '--train_dir', gan_synthetic_data.joinpath('train'),
        '--mrcnn_path', MRCNN_CHECKPOINTS,
        '--val_dir', gan_synthetic_data.joinpath('val'),
        '--n_epochs', NUMBER_OF_EPOCHS_MRCNN,
        '--name', Application_Name
    ]

    maskrcnn_test_arguments = convert_arguments(maskrcnn_test_arguments)
    print(MRCNN_CHECKPOINTS)
    maskrcnn_test = subprocess.run(["python3", "test_model.py"]+maskrcnn_test_arguments,
                                         check=True, capture_output=False, text=True)
    print(maskrcnn_test)

# if SHOW_MRCNN:
#     predict_sample_arguments = [
#         '--val_dir', gan_synthetic_data.joinpath('val'),
#         '--mrcnn_path', MRCNN_CHECKPOINTS,
#         '--name', Application_Name
#     ]
#     predict_sample_arguments = convert_arguments(predict_sample_arguments)
#     prediction = subprocess.run(["python3", "predict_one_labeled.py"]+predict_sample_arguments,
#                                 check=True, capture_output=False, text=True).stdout
#     print(prediction)