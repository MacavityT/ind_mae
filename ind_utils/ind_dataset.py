import torch
import torch.utils.data as data
import os
import os.path as osp
import numpy as np
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from typing import Union

from PIL import Image, ImageOps
from torchvision.datasets.folder import default_loader
from pathlib import Path
import io
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _pillow2array(img, flag='color', channel_order='bgr'):
    """Convert a pillow image to numpy array.

    Args:
        img (:obj:`PIL.Image.Image`): The image loaded using PIL
        flag (str): Flags specifying the color type of a loaded image,
            candidates are 'color', 'grayscale' and 'unchanged'.
            Default to 'color'.
        channel_order (str): The channel order of the output image array,
            candidates are 'bgr' and 'rgb'. Default to 'bgr'.

    Returns:
        np.ndarray: The converted numpy array
    """
    channel_order = channel_order.lower()
    if channel_order not in ['rgb', 'bgr']:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == 'unchanged':
        array = np.array(img)
        if array.ndim >= 3 and array.shape[2] >= 3:  # color image
            array[:, :, :3] = array[:, :, (2, 1, 0)]  # RGB to BGR
    else:
        # Handle exif orientation tag
        if flag in ['color', 'grayscale']:
            img = ImageOps.exif_transpose(img)
        # If the image mode is not 'RGB', convert it to 'RGB' first.
        if img.mode != 'RGB':
            if img.mode != 'LA':
                # Most formats except 'LA' can be directly converted to RGB
                img = img.convert('RGB')
            else:
                # When the mode is 'LA', the default conversion will fill in
                #  the canvas with black, which sometimes shadows black objects
                #  in the foreground.
                #
                # Therefore, a random color (124, 117, 104) is used for canvas
                img_rgba = img.convert('RGBA')
                img = Image.new('RGB', img_rgba.size, (124, 117, 104))
                img.paste(img_rgba, mask=img_rgba.split()[3])  # 3 is alpha
        if flag in ['color', 'color_ignore_orientation']:
            array = np.array(img)
            if channel_order != 'rgb':
                array = array[:, :, ::-1]  # RGB to BGR
        elif flag in ['grayscale', 'grayscale_ignore_orientation']:
            img = img.convert('L')
            array = np.array(img)
        else:
            raise ValueError(
                'flag must be "color", "grayscale", "unchanged", '
                f'"color_ignore_orientation" or "grayscale_ignore_orientation"'
                f' but got {flag}')
    return array


class IndustryDataset(data.Dataset):
    """
    Dataset of Industry, including of several different datasets, 
    loading from .txt file.
    """

    def __init__(
        self,
        root: str,
        ann_file: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
    ):
        self.root = root
        self.ann_file = ann_file
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.samples = self.load_annotations()
        self.imgs = self.samples

    def load_annotations(self):
        assert isinstance(self.ann_file, str)
        if not osp.isabs(self.ann_file):
            self.ann_file = osp.join(self.root, self.ann_file)
        with open(self.ann_file) as f:
            samples = [osp.join(self.root, x.strip()) for x in f.readlines()]
        return samples

    def get_imgbytes(self, filepath: Union[str, Path]) -> bytes:
        """Read data from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes: Expected bytes object.
        """
        with open(filepath, 'rb') as f:
            value_buf = f.read()
        return value_buf

    def imfrombytes(self,
                    content,
                    flag='color',
                    channel_order='bgr',
                    backend=None):
        with io.BytesIO(content) as buff:
            img = Image.open(buff)
            img = _pillow2array(img, flag, channel_order)
        return img

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[index]
        file = path[len(self.root):]
        if file[0] == '/':
            file = file[1:]
        # sample = self.loader(path)
        try:
            img_bytes = self.get_imgbytes(path)
            img = self.imfrombytes(img_bytes,
                                   flag='color',
                                   channel_order='rgb')
            sample = Image.fromarray(img, mode='RGB')
        except Exception as e:
            print(f'Index {index}, Filename {file} " : {repr(e)}')

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        target = file  # self-supervise, not need target
        return sample, target

    def __len__(self) -> int:
        return len(self.samples)