from tkinter import Image
import torch
import numpy as np
import cv2
import os
import os.path as osp
import torchvision.transforms as transforms
from ind_utils.ind_dataset import IndustryPretrainDataset
from torchvision import get_image_backend
from torchvision.datasets.folder import default_loader
from tqdm import trange
from PIL import Image, ImageOps
from util.crop import RandomResizedCrop

root = '/root/ty_room/IndDatasetsSplit'
# industry dataset
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.2),
                                 interpolation=3),  # 3 is bicubic
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_val = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset_train = IndustryPretrainDataset(root=root,
                                        ann_file='train.txt',
                                        transform=transform_train)

print(dataset_train)

bar = trange(len(dataset_train))
count = 0
for i in range(len(dataset_train)):

    sample, target = dataset_train[i]
    if sample.shape != (3, 224, 224):
        print("shape wrong: {}, name: {}".format(sample.shape, target))

    bar.set_description('img name: {}'.format(target))
    bar.update()