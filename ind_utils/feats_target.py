import cv2
import numpy as np
from skimage import feature, exposure


class HOGTarget:

    def __init__(self, norm=False, gamma=1.5):
        self.norm = norm
        self.gamma = gamma

    def _img_norm():
        pass

    def get_hog_map(self, target):

        if self.norm:
            target = np.power(target / float(np.max(target)), self.gamma)

        fd = feature.hog(target,
                         orientations=9,
                         pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2),
                         channel_axis=2 if len(target.shape) > 2 else None,
                         visualize=False)

        # Rescale histogram for better display
        # hog_image_rescaled = exposure.rescale_intensity(hog_image,
        #                                                 in_range=(0, 10))
        # cv2.imwrite('/home/taiyan/hog_img.jpg', hog_image)
        # cv2.imwrite('/home/taiyan/hog_image_rescaled.jpg', hog_image_rescaled)
        return fd

    def __call__(self, target):
        feats = self.get_hog_map(target)
        return feats
