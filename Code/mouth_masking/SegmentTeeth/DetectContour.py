import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from copy import deepcopy
import os


def preprocess(img: np.ndarray):
    """
    Preprocess the image for contour detection
    """
    img_copy = deepcopy(img)
    location = np.where((img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0))
    img_copy[location[0], location[1]] = (255, 255, 255)
    return img_copy


def detect_contour(img: np.ndarray):
    """
    Detect the contour of the image
    """

    img_preprocess = preprocess(img)

    # Canny Edge Detection
    threshold1 = 200
    threshold2 = 100
    img_contour = cv2.Canny(img_preprocess, threshold1, threshold2)

    kernel = np.ones((2, 2), np.uint8)
    img_contour = cv2.dilate(img_contour, kernel, iterations=1)
    img_contour = np.flip(cv2.dilate(np.flip(img_contour), kernel, iterations=1))

    img_contour = Image.fromarray(cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB))
    return img_contour


def masking_mouth(img: np.ndarray, mask: np.ndarray):
    """
    Mask the mouth of the image
    """
    mask_mouth = np.where((mask[:, :, 0] == 255) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 255), 1., 0.)
    mask_mouth = np.expand_dims(mask_mouth, -1)

    img_mouth = mask_mouth * img
    return img_mouth
