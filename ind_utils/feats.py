import cv2
import numpy as np
from skimage import feature, exposure
from .misc import run_time
import torch


class BaseTarget:

    @property
    def feats_len(self):
        raise NotImplementedError()


class HOGTarget(BaseTarget):

    def __init__(self,
                 img_size=(224, 224),
                 patch_size=16,
                 norm=False,
                 gamma=1.5,
                 hog_params=dict(orientations=9,
                                 pixels_per_cell=(8, 8),
                                 cells_per_block=(2, 2),
                                 multichannel=True,
                                 visualize=True)):
        self.img_size = img_size
        self.patch_size = patch_size,
        self.norm = norm
        self.gamma = gamma
        self.hog_params = hog_params

    @property
    def feats_len(self):
        h, w = self.img_size
        h_cell, w_cell = self.hog_params['pixels_per_cell']
        h_block, w_block = self.hog_params['cells_per_block']
        orientations = self.hog_params['orientations']

        h_cell_num = int(h // h_cell)
        w_cell_num = int(w // w_cell)
        h_block_num = h_cell_num - h_block + 1
        w_block_num = w_cell_num - w_block + 1
        block_num = h_block_num * w_block_num

        cell_value = orientations * h_block * w_block
        feats_length = block_num * cell_value

        return feats_length

    def get_hog_map(self, img):
        img = np.transpose(img, (1, 2, 0))  # [h,w,c]
        if self.norm:
            img = np.power(img / float(np.max(img)), self.gamma)

        fd, hog_map = feature.hog(img, **self.hog_params)
        return hog_map

    # @run_time
    def __call__(self, target):
        device = target.device
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()

        feats = []
        b, c, h, w = target.shape
        for i in range(b):
            img = target[i, ...]
            hog_map = self.get_hog_map(img)
            feats.append(hog_map)
        feats = torch.from_numpy(np.array(feats))
        feats = feats.unsqueeze(1).cuda(device=device)  # [b, 1, h, w]
        return feats
