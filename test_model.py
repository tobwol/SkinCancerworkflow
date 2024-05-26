#!/usr/bin/env python3
from time import sleep
from torch.utils.data import DataLoader, Dataset
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm
from natsort import natsorted
import warnings
import matplotlib.cbook
from pathlib import Path
import argparse
import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import csv
class SkincancerDataset(Dataset):

    def __init__(self, root, transforms=None):  # transforms
        self.root = root

        # self.transforms = transforms
        self.transforms = []
        if transforms != None:
            self.transforms.append(transforms)
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(natsorted(os.listdir(os.path.join(root, "images"))))
        # print('imgs file names:', self.imgs)
        self.masks = list(natsorted(os.listdir(os.path.join(root, "masks"))))
        # print('masks file names:', self.masks)
        # images = []
        # masks = []
        # image_names = [f.name for f in os.scandir(root)]
        # image_names = natsorted(image_names)
        # for image_name in image_names:
        #   if image_name.endswith('_anno.bmp'):
        #     masks.append(image_name)
        #   else:
        #     images.append(image_name)
        # print(images)
        # print(masks)
        # print('Length images:', len(images))
        # print('Length masks:', len(masks))
        # self.imgs = images
        # self.masks = masks
        # print('Length images:', len(self.imgs))
        # print('Length masks:', len(self.masks))

    def __getitem__(self, idx):
        # idx sometimes goes over the nr of training images, add logic to keep it lower
        if idx >= 80:
            idx = np.random.randint(80, size=1)[0]
        # print("idx:", idx)

        # load images ad masks
        # print('idx:', idx)
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        # print(img_path)
        # print('img_path', self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        # print(mask_path)
        # print('mask_path', self.masks[idx])
        # print('img_path name:', self.imgs[idx])
        # img_path = os.path.join(self.root, self.imgs[idx])
        # print('mask_path name:', self.masks[idx])
        # mask_path = os.path.join(self.root, self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB, because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path).convert("L")
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        # print(num_objs)
        boxes = []
        # print(num_objs)
        for i in range(num_objs):
            pos = np.where(masks[i])
            # print('pos', pos)
            # print(pos)
            xmin = np.min(pos[1] - 1)
            # print("xmin:", xmin)
            xmax = np.max(pos[1] + 1)
            # print("xmax:", xmax)
            ymin = np.min(pos[0] - 1)
            # print("ymin:", ymin)
            ymax = np.max(pos[0] + 1)
            # print("ymax:", ymax)
            # print('box: ', [xmin, ymin, xmax, ymax])
            # Check if area is larger than a threshold
            # A = abs((xmax - xmin) * (ymax - ymin))
            # print('x', (xmax - xmin))
            # print('y', (ymax - ymin))
            # print('area', A)
            # if A < 9:
            #     # print('Nr before deletion:', num_objs)
            #     # obj_ids = np.delete(obj_ids, [i])
            #     # # print('Area smaller than 5! Box coordinates:', [xmin, ymin, xmax, ymax])
            #     # print('Nr after deletion:', len(obj_ids))
            #     # continue
            #     # xmax=xmax+5
            #     # ymax=ymax+5
            #     pass
            #     print('PASSED')
            # else:
            boxes.append([xmin, ymin, xmax, ymax])

        # print('nr boxes is equal to nr ids:', len(boxes)==len(obj_ids))
        num_objs = len(obj_ids)
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # image_id = torch.tensor([idx])
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # # suppose all instances are not crowd
        # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        for i in self.transforms:
            img = i(img)

        target = {}

        if len(boxes) == 0:
            boxes = torch.zeros(1, 4)
            boxes[2] = img.size[1]
            boxes[3] = img.size[2]
        if len(labels) == 0:
            labels = torch.zeros(1)
        if len(masks) == 0:
            masks = torch.zeros(img.shape)

        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["labels"] = labels  # Not sure if this is needed

        # trans = transforms.ToTensor()
        # img = trans(img)
        # transform = []
        # transform.append(transforms.RandomHorizontalFlip(0.5))
        # transform.append(transforms.ToTensor())
        # transform = Compose(transform)
        # img, target = transform(img, target)

        # if self.transforms is not None:
        #   # for t in self.transforms:
        #   #   img = t(img)
        #   #   target = t(target)
        #     # print(img.size)
        #     img, target = self.transforms(img, target)

        return img.double(), target
    
    def __len__(self):
        return len(self.imgs)
    
def latest_model(application):
    sp = application
    largest = 0
    model_names = os.listdir(os.path.join(sp))
    for i in model_names:
        nr = int(re.findall(r'\d+', i)[0])
        # print(nr)
        if nr > largest:
            largest = nr
    return largest, Path(sp)

def calculate_iou(y_true, y_pred):
    """
    Calculate Intersection over Union (IoU) score.

    Parameters:
    - y_true: Ground truth mask (binary mask, 0 for background, 1 for foreground)
    - y_pred: Predicted mask (binary mask, 0 for background, 1 for foreground)

    Returns:
    - IoU score
    """
    intersection = np.sum(np.logical_and(y_true, y_pred))
    union = np.sum(np.logical_or(y_true, y_pred))

    iou_score = intersection / union if union > 0 else 0

    return iou_score

def calculate_f1_score(y_true, y_pred):
    """
    Calculate the F1 score.

    Parameters:
    - y_true: Ground truth mask (binary mask, 0 for background, 1 for foreground)
    - y_pred: Predicted mask (binary mask, 0 for background, 1 for foreground)

    Returns:
    - F1 score
    """
    # Calculate True Positives, False Positives, and False Negatives
    tp = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    fp = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    fn = np.sum(np.logical_and(y_true == 1, y_pred == 0))

    # Calculate Precision, Recall, and F1 Score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1_score


if __name__ == '__main__':
    print("hello Test")
    # ------------------------------- Run Model ----------------------------------------
    # set the following 3 variables
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', dest='train_dir', action='store', help='Path to the Training Data')
    parser.add_argument('--mrcnn_path', dest='checkpoints_dir', action='store', help='Path to the Saved Models')
    parser.add_argument('--val_dir', dest='val_dir', action='store', help='Path to the Validation Data')
    parser.add_argument('--n_epochs', dest='n_epochs', action='store', help='Number of Epochs')
    parser.add_argument('--name', dest='name', action='store', help='Name of CV Application')

    arguments = parser.parse_args()
    print(arguments)
    train_folder = 'CV_DATA/MASKRCNN_training_data'
    test_folder = 'test'
    model_application = 'Skincancer'
    model_save_path = arguments.checkpoints_dir
    
    # ignore warnings
    warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)


    # Check if GPU is available
    if torch.cuda.is_available():
        print("Running on GPU")
    else:
        print("Running on CPU")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_test = test_folder
    dataset_test = SkincancerDataset(root_test, transforms=torchvision.transforms.ToTensor()) # get_transform(train=True)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=lambda x:list(zip(*x)))
    num_classes = 2

    # load an instance segmentation model pre-trained on COCO
    # choose from maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2, fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True) # weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    # ------------------------------- Training --------------------------------------
    # Load previously trained model
    try:
        print("Searching for the trained model...")
        # List all files in the model_save_path directory
        model_files = os.listdir(model_save_path)
        # Filter out non-model files (if any)
        model_files = [f for f in model_files if f.endswith(".pt")]

        # Check if at least one model file exists
        if not model_files:
            raise FileNotFoundError("No model files found in the specified path.")

        # Assuming there's only one model file, select the first one
        model_file = model_files[0]
        model_path = os.path.join(model_save_path, model_file)

        # Load the model
        model.load_state_dict(torch.load(model_path))
        print(model_file)
        print("Model loaded successfully!")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print("Error loading model:", e)
    # Look at some images and predicted bbox's after training
    images, targets = next(iter(data_loader_test))
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    model = model.double()
    model.eval()
    output = model(images)

    # with torch.no_grad():
    #     view(images, output, 1)


    for i in range(1):
        # out = output[i]['scores'].to('cpu')
        # out = out.detach().numpy()
        for j in range(len(output[i]['scores'])):
            if j < 0.7:
                output[i]['boxes'][j] = torch.Tensor([0, 0, 0, 0])

    # View a sample of masks predicted on test data
    # view_mask(targets, output, n=1)

    # ------------------------------- Results --------------------------------------

    #Get IoU score for whole test set
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Assuming you have a function 'calculate_iou' defined somewhere

IoU_scores_list = []
F1_score_list = []
skipped = 0
for images, targets in tqdm(data_loader_test):
    images = [image.to(device) for image in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    model.eval()
    with torch.no_grad():
        output = model(images)
    threshold = 0.7
    target_im = targets[0]['masks'][0].cpu().detach().numpy()
    for k in range(len(targets[0]['masks'])):
        target_im2 = targets[0]['masks'][k].cpu().detach().numpy()
        target_im2[target_im2 > threshold] = 1
        target_im2[target_im2 < threshold] = 0
        target_im = target_im + target_im2

    target_im[target_im > threshold] = 1
    target_im[target_im < threshold] = 0
    target_im = target_im.astype('int64')

    output_im = output[0]['masks'][0][0, :, :].cpu().detach().numpy()
    for k in range(len(output[0]['masks'])):
        output_im2 = output[0]['masks'][k][0, :, :].cpu().detach().numpy()
        output_im2[output_im2 > threshold] = 1
        output_im2[output_im2 < threshold] = 0
        output_im = output_im + output_im2

    output_im[output_im > threshold] = 1
    output_im[output_im < threshold] = 0
    output_im = output_im.astype('int64')

    # Calculate IoU
    if target_im.shape != output_im.shape:
        skipped += 1
        continue

    IoU_score = calculate_iou(target_im, output_im)
    IoU_scores_list.append(IoU_score)
    F1_score = calculate_f1_score(target_im, output_im)
    F1_score_list.append(F1_score)
    # Plot original image and predicted mask
    original_image = images[0].cpu().permute(1, 2, 0).numpy()
    predicted_mask = output_im
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask, cmap='binary')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.show()

print('----------------- STATISTICS -----------------')
print('mean IoU score for test set:', np.mean(IoU_scores_list))
print('STD IoU score for test set:', np.std(IoU_scores_list))
print('mean F1 score for test set:', np.mean(F1_score_list))
print('STD F1 score for test set:', np.std(F1_score_list))