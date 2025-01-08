import numpy as np
import cv2
import utilities as ut
import matplotlib.pyplot as plt
import fill
import skimage
from typing import Tuple


def kasa(image, min_radius=5, pixel_color_lower_then=10) -> Tuple[int, int, float]:
    """Find dark pixels below pixel_color_lower_then (10 is default). Biggest region is returned.

    Return
    ------
    (center, radius, compactness): Tuple(int, int, float)
    """
    bw = np.zeros(image.shape, np.uint8)
    bw[np.where(image < image.min() + pixel_color_lower_then)] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)  # morfological opening! remove small regions
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)  # morfological closing! fill region
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)  # morfological opening! remove small regions
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)  # morfological closing! fill region
    a, contour, hier = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    iscircle = []
    for i, boundary in enumerate(contour):
        boundary = np.squeeze(boundary)
        if len(boundary) < 12:
            continue
        temp = np.zeros_like(image)
        cv2.drawContours(temp, contour, i, color=255, thickness=-1)
        reg = skimage.measure.regionprops(temp)[0]
        center, radius = (reg.centroid[1], reg.centroid[0]), reg.equivalent_diameter / 2
        if not ut.inimage(center, radius, image, min(image.shape) / 20):
            continue
        iscircle.append([ut.roundToInt(center), int(radius), ut.get_contour_compactness(boundary)])
    # [surface, center, radius, compactness]
    iscircle = sorted(iscircle, key=lambda a: a[1], reverse=True)
    for c in iscircle:
        return tuple(ut.roundToInt(c[0])), c[1],  c[2]
    return None, 0, 5
