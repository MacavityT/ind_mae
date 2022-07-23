import os
import copy
from PIL import Image
import torchvision.transforms as transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from . import sewer_datasets

DS_MODE_MAP = {
    "e2e": dict(dataset="SewerMultiLabelDataset", onlyDefects=False),
    "defect": dict(dataset="SewerMultiLabelDataset", onlyDefects=True),
    "binary": dict(dataset="SewerBinaryDataset"),
    "binaryrelevance": dict(dataset="SewerBinaryRelevanceDataset", defect=None)
}


def build_sewer_dataset(args, split):
    # get dataset settings
    ds_mode_map = copy.deepcopy(DS_MODE_MAP)
    ds_params = ds_mode_map[args.ds_mode]
    print(ds_params)
    ds_name = ds_params.pop('dataset')
    if ds_name == 'binaryrelevance':
        assert args.br_defect is not None, "Training mode is 'binary_relevance', but no 'br_defect' was stated"
        ds_params['defect'] = args.br_defect

    ann_root = os.path.join(args.data_path, 'annotations')
    img_root = os.path.join(args.data_path, 'images', split.lower())

    transform = build_sewer_transform(True if split == 'Train' else False,
                                      args)
    dataset = sewer_datasets.__dict__[ds_name](annRoot=ann_root,
                                               imgRoot=img_root,
                                               transform=transform,
                                               **ds_params)
    print(dataset)
    return dataset


def create_sewer_transform(is_train, args):
    img_size = args.input_size
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1,
                                   contrast=0.1,
                                   saturation=0.1,
                                   hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.523, 0.453, 0.345],
                                 std=[0.210, 0.199, 0.154])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.523, 0.453, 0.345],
                                 std=[0.210, 0.199, 0.154])
        ])
    return transform


def build_sewer_transform(is_train, args):
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
