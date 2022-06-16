from logging import PlaceHolder
import cv2
import numpy as np
from skimage import feature, exposure

# img = cv2.imread('gamma.jpg', 0)
# img2 = np.power(img / float(np.max(img)), 1.5)

# # Read image
# img = cv2.imread('runner.jpg')
# img = np.float32(img) / 255.0  # 归一化

# # 计算x和y方向的梯度
# gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
# gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

# # 计算合梯度的幅值和方向（角度）
# mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

path = '/home/taiyan/penguin.jpg'
image = cv2.imread(path)
# image = np.power(image / float(np.max(image)), 1.5)
# w = int(image.shape[1] / 4)
# h = int(image.shape[0] / 4)
# image = cv2.resize(image, (w, h))
fd, hog_image = feature.hog(image,
                            orientations=9,
                            pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2),
                            channel_axis=2 if len(image.shape) > 2 else None,
                            visualize=True)

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
# cv2.imwrite('/home/taiyan/hog_img.jpg', hog_image)
# cv2.imwrite('/home/taiyan/hog_image_rescaled.jpg', hog_image_rescaled)

# cv2.imshow('img', image)
# cv2.imshow('hog', hog_image_rescaled)
# cv2.waitKey(0) == ord('q')


class GTFeatsGenerator:

    def __init__(self, feats):
        PlaceHolder

    def __call__(self):
        pass