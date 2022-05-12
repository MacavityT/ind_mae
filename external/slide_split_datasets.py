import os
import os.path as osp
import numpy as np
# from scipy.misc import imread
import cv2
from tqdm import trange
import multiprocessing

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
        img = cv2.imread(img_id, cv2.IMREAD_COLOR)
        h_img, w_img, _ = img.shape

        # slide split image as patches
        h_patch, w_patch = patch_size
        h_stride, w_stride = stride
        h_grids = max(h_img - h_patch + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_patch + w_stride - 1, 0) // w_stride + 1

        index = 0
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_patch, h_img)
                x2 = min(x1 + w_patch, w_img)
                y1 = max(y2 - h_patch, 0)
                x1 = max(x2 - w_patch, 0)
                patch_img = img[y1:y2, x1:x2, :]

                index += 1
                patch_img_name = img_name + '_patch' + str(index) + '.jpg'
                img_id_new = osp.join(
                    img_path_new, patch_img_name)[len(DATASET_PATH_NEW) + 1:]
                new_ids.append(img_id_new)
                patch_img_save_path = osp.join(DATASET_PATH_NEW, img_id_new)
                cv2.imwrite(patch_img_save_path, patch_img)

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
