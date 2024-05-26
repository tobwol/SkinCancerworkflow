#!/usr/bin/env python3
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

class ImageComposition:
    """ Composes images together in random ways, applying transformations to the foreground to create a synthetic
        combined image.
    """

    def __init__(self,count):
        print("initalized")
        self.allowed_output_types = ['.png', '.jpg', '.jpeg']
        self.allowed_background_types = ['.png', '.jpg', '.jpeg']
        self.zero_padding = 8  # 00000027.png, supports up to 100 million images
        self.max_foregrounds = 3
        self.mask_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(0, self.max_foregrounds)]
        self.count = count
        print(self.count)
        assert len(self.mask_colors) >= self.max_foregrounds, 'length of mask_colors should be >= max_foregrounds'
    def _validate_and_process_args(self, args):
        # Validates input arguments and sets up class variables
        # Args:
        #     args: the ArgumentParser command line arguments

        self.silent = args.silent

        # Validate the count
        assert args.count > 0, 'count must be greater than 0'
        self.count = args.count

        # Validate the width and height
        assert args.width >= 64, 'width must be greater than 64'
        self.width = args.width
        assert args.height >= 64, 'height must be greater than 64'
        self.height = args.height

        # Validate and process the output type
        if args.output_type is None:
            self.output_type = '.jpg'  # default
        else:
            if args.output_type[0] != '.':
                self.output_type = f'.{args.output_type}'
            assert self.output_type in self.allowed_output_types, f'output_type is not supported: {self.output_type}'

        # Validate and process output and input directories
        self._validate_and_process_output_directory(args)
        self._validate_and_process_input_directory(args)

    def _validate_and_process_output_directory(self, args):
        self.output_dir = Path(args.output_dir)
        self.images_output_dir = self.output_dir / 'images'
        self.masks_output_dir = self.output_dir / 'masks'

        # Create directories
        try:
            self.output_dir.mkdir(exist_ok=False)
            self.images_output_dir.mkdir(exist_ok=False)
            self.masks_output_dir.mkdir(exist_ok=False)
        except FileExistsError:
            shutil.rmtree(self.output_dir)
            self.output_dir.mkdir(exist_ok=False)
            self.images_output_dir.mkdir(exist_ok=False)
            self.masks_output_dir.mkdir(exist_ok=False)

        if not self.silent:
            # Check for existing contents in the images directory
            for _ in self.images_output_dir.iterdir():
                # We found something, check if the user wants to overwrite files or quit
                should_continue = input('output_dir is not empty, files may be overwritten.\nContinue (y/n)? ').lower()
                if should_continue != 'y' and should_continue != 'yes':
                    quit()
                break

    def _validate_and_process_input_directory(self, args):
        self.input_dir = Path(args.input_dir)
        assert self.input_dir.exists(), f'input_dir does not exist: {args.input_dir}'

        for x in self.input_dir.iterdir():
            if 'foregrounds' in x.name:
                self.foregrounds_dir = x
            elif 'backgrounds' in x.name:
                self.backgrounds_dir = x

        assert self.foregrounds_dir is not None, 'foregrounds sub-directory was not found in the input_dir'
        assert self.backgrounds_dir is not None, 'backgrounds sub-directory was not found in the input_dir'

        self._validate_and_process_foregrounds()
        self._validate_and_process_backgrounds()

    def _validate_and_process_foregrounds(self):
        # Validates input foregrounds and processes them into a foregrounds dictionary.
        # Expected directory structure:
        # + foregrounds_dir
        #     + super_category_dir
        #         + category_dir
        #             + foreground_image.png

        self.foregrounds_dict = dict()

        for super_category_dir in self.foregrounds_dir.iterdir():
            if not super_category_dir.is_dir():
                warnings.warn(
                    f'file found in foregrounds directory (expected super-category directories), ignoring: {super_category_dir}')
                continue

            # This is a super category directory
            for category_dir in super_category_dir.iterdir():
                if not category_dir.is_dir():
                    warnings.warn(
                        f'file found in super category directory (expected category directories), ignoring: {category_dir}')
                    continue

                # This is a category directory
                print(category_dir)
                for image_file in category_dir.iterdir():
                    if not image_file.is_file():
                        warnings.warn(f'a directory was found inside a category directory, ignoring: {str(image_file)}')
                        continue
                    if image_file.suffix != '.png':
                        warnings.warn(f'foreground must be a .png file, skipping: {str(image_file)}')
                        continue

                    # Valid foreground image, add to foregrounds_dict
                    super_category = super_category_dir.name
                    category = category_dir.name

                    if super_category not in self.foregrounds_dict:
                        self.foregrounds_dict[super_category] = dict()

                    if category not in self.foregrounds_dict[super_category]:
                        self.foregrounds_dict[super_category][category] = []

                    self.foregrounds_dict[super_category][category].append(image_file)

        assert len(self.foregrounds_dict) > 0, 'no valid foregrounds were found'

    def _validate_and_process_backgrounds(self):
        self.backgrounds = []
        for image_file in self.backgrounds_dir.iterdir():
            if not image_file.is_file():
                warnings.warn(f'a directory was found inside the backgrounds directory, ignoring: {image_file}')
                continue

            if image_file.suffix not in self.allowed_background_types:
                warnings.warn(
                    f'background must match an accepted type {str(self.allowed_background_types)}, ignoring: {image_file}')
                continue

            # Valid file, add to backgrounds list
            self.backgrounds.append(image_file)

        assert len(self.backgrounds) > 0, 'no valid backgrounds were found'

    def _generate_images(self):
        # Generates a number of images and creates segmentation masks, then
        # saves a mask_definitions.json file that describes the dataset.

        print(f'Generating {self.count} images with masks...')


        # Create all images/masks (with tqdm to have a progress bar)
        for i in tqdm(range(self.count)):
            # Randomly choose a background
            background_path = random.choice(self.backgrounds)

            num_foregrounds = random.randint(2, self.max_foregrounds)
            foregrounds = []
            for fg_i in range(num_foregrounds):
                # Randomly choose a foreground
                super_category = random.choice(list(self.foregrounds_dict.keys()))
                category = random.choice(list(self.foregrounds_dict[super_category].keys()))
                foreground_path = random.choice(self.foregrounds_dict[super_category][category])

                # Get the color
                mask_rgb_color = self.mask_colors[fg_i]

                foregrounds.append({
                    'super_category': super_category,
                    'category': category,
                    'foreground_path': foreground_path,
                    'mask_rgb_color': mask_rgb_color
                })

            # Compose foregrounds and background
            composite, mask = self._compose_images(foregrounds, background_path)

            # Create the file name (used for both composite and mask)
            save_filename = f'{i:0{self.zero_padding}}'  # e.g. 00000023.jpg

            # # # ONLY NEED THE MASK TO BE SAVED SINCE MASKS ARE USED IN GAN OUTPUTS
            # # Save composite image to the images sub-directory
            composite_filename = f'{save_filename}{self.output_type}'  # e.g. 00000023.jpg
            composite_path = self.output_dir / 'images' / composite_filename  # e.g. my_output_dir/images/00000023.jpg
            # composite = composite.convert('RGB')  # remove alpha
            # composite.save(composite_path)

            # Save the mask image to the masks sub-directory
            mask_filename = f'{save_filename}.png'  # masks are always png to avoid lossy compression
            mask_path = self.output_dir / 'masks' / mask_filename  # e.g. my_output_dir/masks/00000023.png
            mask.save(mask_path)
            composite.save(composite_path)
        #     color_categories = dict()
        #     for fg in foregrounds:
        #         # Add category and color info
        #         mju.add_category(fg['category'], fg['super_category'])
        #         color_categories[str(fg['mask_rgb_color'])] = \
        #             {
        #                 'category': fg['category'],
        #                 'super_category': fg['super_category']
        #             }

        #     # Add the mask to MaskJsonUtils
        #     mju.add_mask(
        #         composite_path.relative_to(self.output_dir).as_posix(),
        #         mask_path.relative_to(self.output_dir).as_posix(),
        #         color_categories
        #     )

        # # Write masks to json
        # mju.write_masks_to_json()

    def _compose_images(self, foregrounds, background_path):
        # Composes a foreground image and a background image and creates a segmentation mask
        # using the specified color. Validation should already be done by now.
        # Args:
        #     foregrounds: a list of dicts with format:
        #       [{
        #           'super_category':super_category,
        #           'category':category,
        #           'foreground_path':foreground_path,
        #           'mask_rgb_color':mask_rgb_color
        #       },...]
        #     background_path: the path to a valid background image
        # Returns:
        #     composite: the composed image
        #     mask: the mask image

        # Open background and convert to RGBA
        background = Image.open(background_path)
        background = background.convert('RGBA')

        # get background size
        bg_width, bg_height = background.size
        max_crop_x_pos = bg_width
        max_crop_y_pos = bg_height

        # check that foreground is smaller than background
        assert max_crop_x_pos >= 0, f'desired width, {self.width}, is greater than background width, {bg_width}, for {str(background_path)}'
        assert max_crop_y_pos >= 0, f'desired height, {self.height}, is greater than background height, {bg_height}, for {str(background_path)}'
        # crop_x_pos = 0  # random.randint(0+int(bg_width/4), max_crop_x_pos)
        # crop_y_pos = 0  # random.randint(0+int(bg_height/4), max_crop_y_pos)
        # composite = background.crop((crop_x_pos, crop_y_pos, crop_x_pos + self.width, crop_y_pos + self.height))
        src_image_size = background.size
        composite = background.resize((self.width, self.height))
        composite_mask = Image.new('RGB', composite.size, 0)

        position_list_x = list()
        position_list_y = list()
        # print('NEXT IMAGE')
        for fg in foregrounds:
            # get foreground path and perform transformations
            fg_path = fg['foreground_path']
            fg_image = self._transform_foreground(fg, fg_path)

            scale = tuple(dimA/dimB for dimA, dimB in zip(fg_image.size, src_image_size))

            x = scale[0]*self.width
            y = scale[1]*self.height

            if x < 3 or y < 3:
                x = fg_image.size[0]
                y = fg_image.size[1]

            conformed_size = (int(x), int(y))
            fg_image = fg_image.resize(conformed_size)

            try:
                # open_cv_image = numpy.array(fg_image)
                # from scipy import sparse
                # remove_index = list()
                # for i in open_cv_image:
                #     # m = sparse.csr_matrix(i)
                #     # if sparse.issparse(m) is True:
                #     #     # print('sparse found')
                #     #     remove_index.append(i)
                #     if numpy.all(i==0):
                #         remove_index.append(i)
                # c = np.setdiff1d(open_cv_image, remove_index)
                # import torchvision.transforms as T
                # transform = T.ToPILImage()
                # fg_image = transform(c)

                # calculate allowable foreground location square

                padding = 40
                min_x_position = int(composite.size[0] * 0.15 + padding/2)
                min_y_position = int(composite.size[1] * 0.15 + padding/2)
                max_x_position = int((composite.size[0] / 2 * sqrt(2)) - fg_image.size[0] - padding)
                max_y_position = int((composite.size[1] / 2 * sqrt(2)) - fg_image.size[1] - padding)
                assert max_x_position >= 0 and max_y_position >= 0, \
                    f'foreground {fg_path} is too big ({fg_image.size[0]}x{fg_image.size[1]}) for the requested output size ({self.width}x{self.height}), check your input parameters'
                # paste_position = (min_x_position + random.randint(0, max_x_position), min_y_position + random.randint(0, max_y_position))
                # # get paste location min and max values
                # min_x = paste_position[0] - 2
                # min_y = paste_position[1] - 2
                # max_x = paste_position[0] + fg_image.size[0] + 2
                # max_y = paste_position[1] + fg_image.size[1] + 2
                #
                # # check if random foreground x location is already taken, pass if true
                #
                # x_list = list(range(min_x, max_x+1))
                # y_list = list(range(min_y, max_y+1))
                #
                # check = any(item in position_list_x for item in x_list) and any(item in position_list_y for item in y_list)

                # check = False
                # iterator = 0

                # while check is True:
                #     if iterator < 500:
                onboarder = True
                skip_foreground = False
                counter = 0
                while onboarder == True:
                    paste_position = (min_x_position + random.randint(0, max_x_position),
                                    min_y_position + random.randint(0, max_y_position))

                    # get paste location min and max values
                    # min_x = paste_position[0]
                    # min_y = paste_position[1]
                    # max_x = paste_position[0] + fg_image.size[0]
                    # max_y = paste_position[1] + fg_image.size[1]

                    # x_list = list(range(min_x, max_x + 1))
                    # y_list = list(range(min_y, max_y + 1))

                            # check = any(item in position_list_x for item in x_list) and any(
                            #     item in position_list_y for item in y_list)

                        #     iterator += 1
                        # else:
                        #     break

                    # if check is False:
                        # if successful foreground placement, add to list of taken pixels
                    # for i in range(min_x, max_x+1):
                    #     position_list_x.append(i)
                    # for j in range(min_y, max_y+1):
                    #     position_list_y.append(j)

                    # print(list(range(min_x, max_x+1)))
                    # print('position list x', position_list_x)
                    # print(list(range(min_y, max_y+1)))
                    # print('position list y', position_list_y)

                    # Create a new foreground image as large as the composite and paste it on top
                    new_fg_image = Image.new('RGBA', composite.size, color=(0, 0, 0, 0))
                    new_fg_image.paste(fg_image, paste_position)
                    

                    data_new_image = np.asarray(new_fg_image)
                    data_without_alpha = data_new_image[:,:,:3]
                    composite_data = np.asarray(composite_mask)
                    no_zeros = composite_data[np.where(composite_data==0)]

                    #change comment if you want to have no overlapping of defects
                    #data = np.multiply(data_without_alpha,composite_data)
                    data = 0
                    if (np.count_nonzero(data) == 0):
                        #print(data.shape)
                        onboarder = False
                        break 
                    if counter >= 100:
                        skip_foreground = True
                        print("skip_foreground")
                        break
                    counter += 1
                
                if not skip_foreground:
                    alpha_mask = fg_image.getchannel(3)
                    new_alpha_mask = Image.new('L', composite.size, color=0)
                    new_alpha_mask.paste(alpha_mask, paste_position)
                    composite = Image.composite(new_fg_image, composite, new_alpha_mask)

                    # Grab the alpha pixels above a specified threshold
                    alpha_threshold = 200
                    mask_arr = np.array(np.greater(np.array(new_alpha_mask), alpha_threshold), dtype=np.uint8)
                    uint8_mask = np.uint8(mask_arr)  # This is composed of 1s and 0s

                    # Multiply the mask value (1 or 0) by the color in each RGB channel and combine to get the mask
                    mask_rgb_color = fg['mask_rgb_color']
                    red_channel = uint8_mask * mask_rgb_color[0]
                    green_channel = uint8_mask * mask_rgb_color[1]
                    blue_channel = uint8_mask * mask_rgb_color[2]
                    rgb_mask_arr = np.dstack((red_channel, green_channel, blue_channel))
                    isolated_mask = Image.fromarray(rgb_mask_arr, 'RGB')
                    isolated_alpha = Image.fromarray(uint8_mask * 255, 'L')
                    

                    composite_mask = Image.composite(isolated_mask, composite_mask, isolated_alpha)

                    
            except ValueError:
                print("Value_Error")
                max_x_position = composite.size[0] - fg_image.size[0]
                max_y_position = composite.size[1] - fg_image.size[1]
                # max_x_position = composite.size[0]
                # max_y_position = composite.size[1]
                assert max_x_position >= 0 and max_y_position >= 0, \
                    f'foreground {fg_path} is too big ({fg_image.size[0]}x{fg_image.size[1]}) ' \
                    f'for the requested output size ({self.width}x{self.height}), check your input parameters'

                paste_position = (random.randint(10, max_x_position), random.randint(10, max_y_position))

                # Create a new foreground image as large as the composite and paste it on top
                new_fg_image = Image.new('RGBA', composite.size, color=(0, 0, 0, 0))
                new_fg_image.paste(fg_image, paste_position)

                # Extract the alpha channel from the foreground and paste it into a new image the size of the composite
                alpha_mask = fg_image.getchannel(3)
                new_alpha_mask = Image.new('L', composite.size, color=0)
                new_alpha_mask.paste(alpha_mask, paste_position)
                composite = Image.composite(new_fg_image, composite, new_alpha_mask)

                # Grab the alpha pixels above a specified threshold
                alpha_threshold = 200
                mask_arr = np.array(np.greater(np.array(new_alpha_mask), alpha_threshold), dtype=np.uint8)
                uint8_mask = np.uint8(mask_arr)  # This is composed of 1s and 0s

                # Multiply the mask value (1 or 0) by the color in each RGB channel and combine to get the mask
                mask_rgb_color = fg['mask_rgb_color']
                red_channel = uint8_mask * mask_rgb_color[0]
                green_channel = uint8_mask * mask_rgb_color[1]
                blue_channel = uint8_mask * mask_rgb_color[2]
                rgb_mask_arr = np.dstack((red_channel, green_channel, blue_channel))
                isolated_mask = Image.fromarray(rgb_mask_arr, 'RGB')
                isolated_alpha = Image.fromarray(uint8_mask * 255, 'L')

                composite_mask = Image.composite(isolated_mask, composite_mask, isolated_alpha)

    
        #merge masks
        bwimg= composite_mask.convert('L')
        np_image = np.array(bwimg)
        _, img_bw = cv2.threshold(np_image, 2, 255, cv2.THRESH_BINARY)
        np_image = cv2.cvtColor(np_image,cv2.COLOR_GRAY2RGB)
        contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        colors = set()
        while len(colors) < len(contours):
            color = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            if color not in colors:
                colors.add(color)

        colors = list(colors)


        for cnt in range (len(contours)):
            cv2.drawContours(np_image, contours, cnt, colors[cnt], thickness=cv2.FILLED)

        img = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        composite_mask = Image.fromarray(img)

        return composite, composite_mask

    def _transform_foreground(self, fg, fg_path):
        # Open foreground and get the alpha channel
        fg_image = Image.open(fg_path)
        fg_alpha = np.array(fg_image.getchannel(3))
        # assert np.any(fg_alpha == 0), f'foreground needs to have some transparency: {str(fg_path)}'

        # ** Apply Transformations **
        # Rotate the foreground
        angle_degrees = random.randint(0, 359)
        fg_image = fg_image.rotate(angle_degrees, resample=Image.BICUBIC, expand=True)

        # Scale the foreground
        # scale = random.random() * .5 + 1  # Pick something between .5 and 1
        # new_size = (int(fg_image.size[0] * scale), int(fg_image.size[1] * scale))
        #
        # fg_image = fg_image.resize(new_size, resample=Image.BICUBIC)

        # Adjust foreground brightness
        brightness_factor = random.random() * .5 + .75  # Pick something between .7 and 1.1
        enhancer = ImageEnhance.Brightness(fg_image)
        fg_image = enhancer.enhance(brightness_factor)

        # Add any other transformations here...

        return fg_image

    def _create_info(self):
        # A convenience wizard for automatically creating dataset info
        # The user can always modify the resulting .json manually if needed

        if self.silent:
            # SETTING DEFAULT INFO FILE
            # print('Note - USING DEFAULTS: you can always modify the json manually if you need to update this.')
            info = dict()
            info['description'] = 'COCO Based Synthetic Image Dataset for Mask RCNN Model'
            info['url'] = 'SEE FUCHS or NYE LUBRICANTS'
            info['version'] = 'DEFAULT'
            info['contributor'] = 'Script modified by RJB'
            now = datetime.now()
            info['year'] = now.year
            info['date_created'] = f'{now.month:0{2}}/{now.day:0{2}}/{now.year}'

            image_license = dict()
            image_license['id'] = 0

            image_license['name'] = 'Fuchs Lubricants'
            image_license['url'] = 'search for Computer Vision'

            dataset_info = dict()
            dataset_info['info'] = info
            dataset_info['license'] = image_license

            # Write the JSON output file
            output_file_path = Path(self.output_dir) / 'dataset_info.json'
            with open(output_file_path, 'w+') as json_file:
                json_file.write(json.dumps(dataset_info))

            print('Successfully created synthetic images')
            return

        should_continue = input('Would you like to create dataset info json? (y/n) ').lower()
        if should_continue != 'y' and should_continue != 'yes':
            print('No problem. You can always create the json manually.')
            quit()

        # print('Note: you can always modify the json manually if you need to update this.')
        info = dict()
        info['description'] = input('Description: ')
        info['url'] = input('URL: ')
        info['version'] = input('Version: ')
        info['contributor'] = input('Contributor: ')
        now = datetime.now()
        info['year'] = now.year
        info['date_created'] = f'{now.month:0{2}}/{now.day:0{2}}/{now.year}'

        image_license = dict()
        image_license['id'] = 0

        should_add_license = input('Add an image license? (y/n) ').lower()
        if should_add_license != 'y' and should_add_license != 'yes':
            image_license['url'] = ''
            image_license['name'] = 'None'
        else:
            image_license['name'] = input('License name: ')
            image_license['url'] = input('License URL: ')

        dataset_info = dict()
        dataset_info['info'] = info
        dataset_info['license'] = image_license

        # Write the JSON output file
        output_file_path = Path(self.output_dir) / 'dataset_info.json'
        with open(output_file_path, 'w+') as json_file:
            json_file.write(json.dumps(dataset_info))

        print('Successfully created synthetic images')

    # Start here
    def main(self, args):
        print("starting main")
        print(args)
        self._validate_and_process_args(args)
        self._generate_images()
        self._create_info()
        print('Image composition completed.')


def generate(output_dir, count, data, size):
    # GENERATE SYNTHETIC TRAIN DATASET
    synthetic_compositions = ImageComposition()
    synth_args = argparse.Namespace(
        input_dir=data,
        output_dir=output_dir,
        count=count,
        width=size,
        height=size,
        output_type='png',
        silent=True
    )
    synthetic_compositions.main(synth_args)