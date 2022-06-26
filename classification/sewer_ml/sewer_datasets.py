import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
"""
The models can be trained in four different "training modes", based on the type of classifier is wanted:
- training_mode = e2e: An end-to-end classifier is trained on all the data with multi-label annotations.
- training_mode = binary: A binary classifier is trained on all the data with binary defect annotations.
- training_mode = defect: A defect classifier is trained on just the data with defects occuring, with multi-label annotations.
- training_mode = binaryrelevance: A binary classifier is trained on the data with just the defect denoted in the "br_defect" argument.
"""

Labels = [
    "RB",
    "OB",
    "PF",
    "DE",
    "FS",
    "IS",
    "RO",
    "IN",
    "AF",
    "BE",
    "FO",
    "GR",
    "PH",
    "PB",
    "OS",
    "OP",
    "OK",
    "VA",
    "ND",
]


class SewerMultiLabelDataset(Dataset):

    def __init__(
        self,
        annRoot,
        imgRoot,
        split="Train",
        transform=None,
        loader=default_loader,
        onlyDefects=False,
    ):
        super(SewerMultiLabelDataset, self).__init__()
        self.imgRoot = imgRoot
        self.annRoot = annRoot
        self.split = split

        self.transform = transform
        self.loader = default_loader

        self.LabelNames = Labels.copy()
        self.LabelNames.remove("VA")
        self.LabelNames.remove("ND")
        self.onlyDefects = onlyDefects

        self.num_classes = len(self.LabelNames)

        self.loadAnnotations()
        self.class_weights = self.getClassWeights()

    def loadAnnotations(self):
        gtPath = os.path.join(self.annRoot,
                              "SewerML_{}.csv".format(self.split))
        gt = pd.read_csv(
            gtPath,
            sep=",",
            encoding="utf-8",
            usecols=self.LabelNames + ["Filename", "Defect"],
        )

        if self.onlyDefects:
            gt = gt[gt["Defect"] == 1]

        self.imgPaths = gt["Filename"].values
        self.labels = gt[self.LabelNames].values

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        path = self.imgPaths[index]

        img = self.loader(os.path.join(self.imgRoot, path))
        if self.transform is not None:
            img = self.transform(img)

        target = self.labels[index, :]

        return img, target

    def getClassWeights(self):
        data_len = self.labels.shape[0]
        class_weights = []

        for defect in range(self.num_classes):
            pos_count = len(self.labels[self.labels[:, defect] == 1])
            neg_count = data_len - pos_count

            class_weight = neg_count / pos_count if pos_count > 0 else 0
            class_weights.append(np.asarray([class_weight]))
        class_weights = np.asarray(class_weights)
        return torch.as_tensor(class_weights).squeeze()


class SewerMultiLabelDatasetInference(Dataset):

    def __init__(
        self,
        annRoot,
        imgRoot,
        split="Train",
        transform=None,
        loader=default_loader,
        onlyDefects=False,
    ):
        super(SewerMultiLabelDatasetInference, self).__init__()
        self.imgRoot = imgRoot
        self.annRoot = annRoot
        self.split = split

        self.transform = transform
        self.loader = default_loader

        self.LabelNames = Labels.copy()
        self.LabelNames.remove("VA")
        self.LabelNames.remove("ND")
        self.onlyDefects = onlyDefects

        self.num_classes = len(self.LabelNames)

        self.loadAnnotations()

    def loadAnnotations(self):
        gtPath = os.path.join(self.annRoot,
                              "SewerML_{}.csv".format(self.split))
        gt = pd.read_csv(gtPath,
                         sep=",",
                         encoding="utf-8",
                         usecols=["Filename"])

        self.imgPaths = gt["Filename"].values

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        path = self.imgPaths[index]

        img = self.loader(os.path.join(self.imgRoot, path))
        if self.transform is not None:
            img = self.transform(img)

        return img, path


class SewerBinaryRelevanceDataset(Dataset):

    def __init__(
        self,
        annRoot,
        imgRoot,
        split="Train",
        transform=None,
        loader=default_loader,
        defect=None,
    ):
        super(SewerBinaryRelevanceDataset, self).__init__()
        self.imgRoot = imgRoot
        self.annRoot = annRoot
        self.split = split

        self.transform = transform
        self.loader = default_loader

        self.LabelNames = Labels.copy()
        self.LabelNames.remove("VA")
        self.LabelNames.remove("ND")
        self.defect = defect

        assert self.defect in self.LabelNames

        self.num_classes = 1

        self.loadAnnotations()
        self.class_weights = self.getClassWeights()

    def loadAnnotations(self):
        gtPath = os.path.join(self.annRoot,
                              "SewerML_{}.csv".format(self.split))
        gt = pd.read_csv(gtPath,
                         sep=",",
                         encoding="utf-8",
                         usecols=["Filename", self.defect])

        self.imgPaths = gt["Filename"].values
        self.labels = gt[self.defect].values.reshape(self.imgPaths.shape[0], 1)

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        path = self.imgPaths[index]

        img = self.loader(os.path.join(self.imgRoot, path))
        if self.transform is not None:
            img = self.transform(img)

        target = self.labels[index]

        return img, target

    def getClassWeights(self):
        pos_count = len(self.labels[self.labels == 1])
        neg_count = self.labels.shape[0] - pos_count
        class_weight = np.asarray([neg_count / pos_count])

        return torch.as_tensor(class_weight)


class SewerBinaryDataset(Dataset):

    def __init__(self,
                 annRoot,
                 imgRoot,
                 split="Train",
                 transform=None,
                 loader=default_loader):
        super(SewerBinaryDataset, self).__init__()
        self.imgRoot = imgRoot
        self.annRoot = annRoot
        self.split = split

        self.transform = transform
        self.loader = default_loader

        self.num_classes = 1

        self.loadAnnotations()
        self.class_weights = self.getClassWeights()

    def loadAnnotations(self):
        gtPath = os.path.join(self.annRoot,
                              "SewerML_{}.csv".format(self.split))
        gt = pd.read_csv(gtPath,
                         sep=",",
                         encoding="utf-8",
                         usecols=["Filename", "Defect"])

        self.imgPaths = gt["Filename"].values
        self.labels = gt["Defect"].values.reshape(self.imgPaths.shape[0], 1)
        print(self.labels.shape)

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        path = self.imgPaths[index]

        img = self.loader(os.path.join(self.imgRoot, path))
        if self.transform is not None:
            img = self.transform(img)

        target = self.labels[index]

        return img, target

    def getClassWeights(self):
        pos_count = len(self.labels[self.labels == 1])
        neg_count = self.labels.shape[0] - pos_count
        class_weight = np.asarray([neg_count / pos_count])

        return torch.as_tensor(class_weight)


def build_sewer_dataset(args, **kwargs):
    '''
    kwargs = dict(dataset = str,
                  split = str, 
                  defect = str, 
                  onlyDefects = bool)
    '''
    ds = kwargs['dataset']
    split = kwargs['split']
    ann_root = os.path.join(args.data_path, 'annotations')
    img_root = os.path.join(args.data_path, 'images', split.lower())

    transform = build_transform(True if split == 'Train' else False, args)
    dataset = __dict__[ds](annRoot=ann_root,
                           imgRoot=img_root,
                           transform=transform,
                           **kwargs)
    print(dataset)
    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=Image.BICUBIC
                          ),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms

    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor()])

    train = SewerMultiLabelDataset(
        annRoot="/mnt/tmp/SewerML/annotations",
        imgRoot="/mnt/tmp/SewerML/images/train",
        split="Train",
        transform=transform,
    )
    train_defect = SewerMultiLabelDataset(
        annRoot="/mnt/tmp/SewerML/annotations",
        imgRoot="/mnt/tmp/SewerML/images/train",
        split="Train",
        transform=transform,
        onlyDefects=True,
    )
    train_inference = SewerMultiLabelDatasetInference(
        annRoot="/mnt/tmp/SewerML/annotations",
        imgRoot="/mnt/tmp/SewerML/images/train",
        split="Train",
        transform=transform,
    )
    binary_train = SewerBinaryDataset(
        annRoot="/mnt/tmp/SewerML/annotations",
        imgRoot="/mnt/tmp/SewerML/images/train",
        split="Train",
        transform=transform,
    )
    binary_relevance_train = SewerBinaryRelevanceDataset(
        annRoot="/mnt/tmp/SewerML/annotations",
        imgRoot="/mnt/tmp/SewerML/images/train",
        split="Train",
        transform=transform,
        defect="RB",
    )

    print(
        len(train),
        len(train_defect),
        len(train_inference),
        len(binary_train),
        len(binary_relevance_train),
    )
    print(
        train.class_weights,
        train_defect.class_weights,
        binary_train.class_weights,
        binary_relevance_train.class_weights,
    )
