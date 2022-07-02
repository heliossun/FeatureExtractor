import cv2
import skimage.exposure
import numpy as np
from numpy.random import default_rng
import os
from PIL import Image


def add_mask(path):
    img = cv2.imread(path)
    height, width = img.shape[:2]
    mask = generate_mask(height,width)
    result = cv2.add(img, mask)
    im_pil = Image.fromarray(result)
    return im_pil

def generate_mask(height,width):
    seedval = 55
    rng = default_rng(seed=seedval)

    # create random noise image
    noise = rng.integers(0, 255, (height,width), np.uint8, True)

    # blur the noise image to control the size
    blur = cv2.GaussianBlur(noise, (0,0), sigmaX=15, sigmaY=15, borderType = cv2.BORDER_DEFAULT)

    # stretch the blurred image to full dynamic range
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0,255)).astype(np.uint8)

    # threshold stretched image to control the size
    thresh = cv2.threshold(stretch, 175, 255, cv2.THRESH_BINARY)[1]

    # apply morphology open and close to smooth out shapes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.merge([mask,mask,mask])
    return mask
