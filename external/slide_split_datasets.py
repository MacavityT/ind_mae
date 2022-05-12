import os
import os.path as osp
import numpy as np
# from scipy.misc import imread
import cv2
from tqdm import trange
import multiprocessing

from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from typing import Union
from PIL import Image, ImageOps
from pathlib import Path
from PIL import ImageFile
import io

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


def get_imgbytes(filepath: Union[str, Path]) -> bytes:
    """Read data from a given ``filepath`` with 'rb' mode.

    Args:
        filepath (str or Path): Path to read data.

    Returns:
        bytes: Expected bytes object.
    """
    with open(filepath, 'rb') as f:
        value_buf = f.read()
    return value_buf


def imfrombytes(content, flag='color', channel_order='bgr', backend=None):
    with io.BytesIO(content) as buff:
        img = Image.open(buff)
        img = _pillow2array(img, flag, channel_order)
    return img


# configs
ROOT = '/root/ty_room/'
DATASET_PATH_ORIGIN = 'IndDatasets'
DATASET_PATH_NEW = 'IndDatasetsSplit'
IMG_TXT = 'train.txt'

DATASET_PATH_ORIGIN = osp.join(ROOT, DATASET_PATH_ORIGIN)
DATASET_PATH_NEW = osp.join(ROOT, DATASET_PATH_NEW)
IMG_TXT_ORIGIN = osp.join(DATASET_PATH_ORIGIN, IMG_TXT)
IMG_TXT_NEW = osp.join(DATASET_PATH_NEW, IMG_TXT)

if not os.path.exists(DATASET_PATH_NEW):
    os.makedirs(DATASET_PATH_NEW)
EXCLUDE_DATASETS = ['MVTEC_AD', 'MVTEC_D2S', 'MVTEC_LOCO_AD', 'MVTEC_SCREWS']


def _slide_split(ids, stride, patch_size):
    """Get image patches by sliding-window with overlap.

    If h_crop > h_img or w_crop > w_img, the small patch will be used to
    decode without padding.
    """
    tbar = trange(len(ids))
    new_ids = []
    new_labels = []

    for i in tbar:
        # if i == 10:
        #     break

        img_id = ids[i]
        img_name = img_id.split('/')[-1].split('.')[0]
        img_path_origin = osp.abspath(osp.dirname(img_id))
        img_path_new = img_path_origin.replace(DATASET_PATH_ORIGIN,
                                               DATASET_PATH_NEW)
        # read image and object labels
        img_suffix = img_id.split('.')[-1]
        try:
            if img_suffix not in ['tif', 'tiff']:
                img = cv2.imread(img_id, cv2.IMREAD_COLOR)
            else:
                img_bytes = get_imgbytes(img_id)
                img = imfrombytes(img_bytes, flag='color', channel_order='rgb')
        except Exception as e:
            print(f'Image read error :{img_id}')
        assert img is not None
        h_img, w_img, _ = img.shape

        # slide split image as patches
        h_patch, w_patch = patch_size
        h_stride, w_stride = stride
        h_grids = max(h_img - h_patch + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_patch + w_stride - 1, 0) // w_stride + 1

        index = 0
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                index += 1
                patch_img_name = img_name + '_patch' + str(index) + '.jpg'
                img_id_new = osp.join(
                    img_path_new, patch_img_name)[len(DATASET_PATH_NEW) + 1:]
                patch_img_save_path = osp.join(DATASET_PATH_NEW, img_id_new)

                # save file if not exists
                if not osp.exists(patch_img_save_path):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_patch, h_img)
                    x2 = min(x1 + w_patch, w_img)
                    y1 = max(y2 - h_patch, 0)
                    x1 = max(x2 - w_patch, 0)

                    patch_img = img[y1:y2, x1:x2, :]
                    cv2.imwrite(patch_img_save_path, patch_img)
                new_ids.append(img_id_new)

        tbar.set_description('Doing: {}/{}, got {} splited images'.\
            format(i, len(ids), len(new_ids)))

    # with open(PATCH_LABEL_PATH, 'w') as f:
    #     for name, label in zip(new_names, new_labels):
    #         f.write('{} {}\n'.format(name, label))

    return dict(names=new_ids, labels=new_labels)


def main(mode, stride, patch_size):
    with open(IMG_TXT_ORIGIN, 'r') as f:
        ids = []
        for id in f.readlines():
            dataset_name = id.split('/')[0]
            if dataset_name in EXCLUDE_DATASETS:
                continue

            id_abs = osp.join(DATASET_PATH_ORIGIN, id.strip())
            ids.append(id_abs)

            img_path_origin = osp.abspath(osp.dirname(id_abs))
            img_path_new = img_path_origin.replace(DATASET_PATH_ORIGIN,
                                                   DATASET_PATH_NEW)
            if not os.path.exists(img_path_new):
                os.makedirs(img_path_new)

    if mode == 'single-thread':
        ## single thread
        # train set
        dataset = _slide_split(ids, stride, patch_size)
    else:
        ## multi-thread
        cpu_num = multiprocessing.cpu_count()
        # train set
        num_ids_split = len(ids) // cpu_num + 1
        ids_split = []
        for i in range(0, len(ids), num_ids_split):
            e_i = i + num_ids_split
            if e_i > len(ids):
                e_i = len(ids)
            ids_split.append(ids[i:e_i])

        print("Number of cores: {}, images per core: {}".format(
            cpu_num, len(ids_split[0])))
        workers = multiprocessing.Pool(processes=cpu_num)
        processes = []
        for proc_id, ids_set in enumerate(ids_split):
            p = workers.apply_async(_slide_split,
                                    (ids_set, stride, patch_size))
            processes.append(p)

        names = []
        labels = []

        for p in processes:
            dataset_slice = p.get()
            names.extend(dataset_slice['names'])
            labels.extend(dataset_slice['labels'])

        dataset = dict(names=names, labels=labels)

    with open(IMG_TXT_NEW, 'w') as f:
        for name, label in zip(dataset['names'], dataset['labels']):
            # f.write('{} {}\n'.format(name, label))
            f.write(f'{name}\n')


if __name__ == '__main__':
    # mode = 'single-thread'
    mode = 'multi-thread'
    stride = (224, 224)
    patch_size = (224, 224)
    main(mode, stride, patch_size)
