import itertools
import numpy as np
import cv2
import matplotlib.pyplot as plt
from Enums import EyeDiseases
from scipy.ndimage import distance_transform_edt as distance
from enum import Enum
import skimage
import sys
print(sys.version)
print(skimage.__version__)
from skimage.segmentation import _chan_vese, active_contour, morphological_geodesic_active_contour, circle_level_set, inverse_gaussian_gradient, morphological_chan_vese
from skimage.morphology import disk
from sklearn.cluster import spectral_clustering
import starburst as st
from contextlib import ContextDecorator
from timeit import default_timer as time
import skimage
from skimage.measure import label
from skimage.morphology import convex_hull_image
from skimage.util import invert
from skimage.measure import regionprops, points_in_poly
import othersSegmentationMethods as segmentation
from skimage.filters import threshold_otsu
from scipy import signal
from skimage.filters import sobel_v, scharr_v, prewitt_v, sobel_h, scharr_h, prewitt_h
import pandas as pd


def imshow(image, text=""):
    return
    if image.dtype == np.dtype(bool):
        cv2.imshow(text, image.astype(np.double))
    else:
        cv2.imshow(text, image)
    cv2.waitKey(0)


def imshow1(image, text=""):
    return
    if image.dtype == np.dtype(bool):
        cv2.imshow(text, image.astype(np.double))
    else:
        cv2.imshow(text, image)
    cv2.waitKey(0)


def hsv2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def imshow2(image, text=""):
    return
    if image.dtype == np.dtype(bool):
        cv2.imshow(text, image.astype(np.double))
    else:
        cv2.imshow(text, image)
    cv2.waitKey(0)


def imshow3(image, text="", verbose=0, waitkey=True):
    if verbose <= 0:
        return
    if image.dtype == np.dtype(bool):
        cv2.imshow(text, image.astype(np.double))
    else:
        cv2.imshow(text, image)
    return cv2.waitKey(0) if waitkey else None


def polar2cart(theta, r):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return (x, y)


def z2polar(z):
    return (np.abs(z), np.angle(z))


def mousePosition(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(x, y)


def drawImageMouseEvent(I):
    cv2.namedWindow('image')
    cv2.setMouseCallback("image", mousePosition)
    while(1):
        cv2.imshow('image', I)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break


def cv_size(img):
    return tuple(img.shape[1::-1])


def roundToInt(num):
    if isinstance(num, int):
        return int(round(num))
    temp = np.round(num).astype(int)
    return temp


def ceilToInt(num):
    return int(np.ceil(num))


def inimage(center, radius, image, corner_limit=0):
    h, w = image.shape
    [x1, _], [x2, y2] = np.subtract(center, radius), np.add(center, radius)
    # y1 < corner_limit or
    if x1 < corner_limit or y2 > h - corner_limit or x2 > w - corner_limit:
        return False
    return True


def _analyzeCircle(img, centeriris, radius_iris):
    xmin, ymin = np.subtract(centeriris, radius_iris)
    xmax, ymax = np.add(centeriris, radius_iris)
    res = np.multiply(img[ymin:ymax, xmin:xmax], disk(radius_iris * 2))
    imshow(res)
    pass


def euclidean_distance(center1, center2):
    if center1 is None or center2 is None:
        return 1000
    return np.sqrt(np.sum(np.subtract(center1, center2)**2))


def is_cataract_calculate(img, indices, verbose=1):
    tempimg = img[indices]
    mean_ = np.mean(tempimg)
    std = np.std(tempimg)
    if verbose > 0:
        print('cataract mean: ', mean_, '  std: ', std)
    # plt.hist(tempimg, 256, [1, 256])  # arguments are passed to np.histogram
    # plt.show()
    # tempimg = 0
    # imshow1(img, 'combined')
    return mean_ > 35 or std < 4 and mean_ > 20


def get_circle_pixels(img, center, radius, max_value=255):
    return cv2.circle(np.zeros_like(img), tuple(roundToInt(center)), roundToInt(radius), (max_value,) * 3, -1)


def get_circle_pixels_values(img, center, radius):
    return img[np.where(get_circle_pixels(img, center, radius, max_value=1))]


def is_cataract_from_radius(img, center, radius, verbose):
    if radius < 1:
        return False
    indices1 = np.where(img > 100)
    temp = get_circle_pixels(img, center, radius * 0.7)
    temp[indices1] = 0

    return is_cataract_calculate(img, np.where(temp), verbose)


def is_cataract(img, indices, verbose):
    indices1 = np.where(img > 100)
    temp = np.zeros_like(img)
    temp[indices] = 1
    temp[indices1] = 0
    return is_cataract_calculate(img, np.where(temp), verbose)


# for idx, sigma in enumerate([4.0, 8.0, 16.0, 32.0]):
#     s1 = gaussian(img, k * sigma)
#     s2 = gaussian(img, sigma)

#     # multiply by sigma to get scale invariance
#     dog = s1 - s2
#     plt.subplot(2, 3, idx + 2)
#     plt.imshow(dog, cmap='gray')
#     plt.title('DoG with sigma=' + str(sigma) + ', k=' + str(k))
# plt.show()

# for idx, edge in enumerate([sobel_v, scharr_v, prewitt_v, sobel_h, scharr_h, prewitt_h]):
#     plt.subplot(2, 3, idx + 1)
#     plt.imshow(edge(stacked), cmap='gray')
# plt.show()
# s = np.array(list(map(list, zip(range(s.shape[0]), s))))
# from sklearn.mixture import GaussianMixture
# gmm = GaussianMixture(n_components=2, covariance_type='tied').fit(s)
# colors = [['navy', 'c', 'cornflowerblue', 'gold', 'darkorange'][i % 5] for i in gmm.predict(s)]
# plt.scatter(s[:, 0], s[:, 1], c=colors, alpha=0.8)
# plt.show()
def is_arcus_senilis(img, center_pupil, radius_pupil, center_iris_out, radius_iris_out, center_iris_in, radius_iris_in, isarcus=None):
    if not is_iris_diff_arcus(radius_iris_out, radius_iris_in):  # if it is radius in range
        return False
    if not (is_circle_inside(center_pupil, radius_pupil, center_iris_in, radius_iris_in)  #
            and is_circle_inside(center_iris_in, radius_iris_in, center_iris_out, radius_iris_out)):
        return False
    radius_iris_out = int(radius_iris_out * 1.2)
    img, _ = crop_image(img, center_iris_out, radius_iris_out)
    radius_start = roundToInt(radius_iris_in * .8)
    rectangle_img = (im_to_polar(img, radius_iris_out, radius_start=radius_start) * 255.).astype(np.uint8)
    # imshow3(cv2.Canny(rectangle_img, rectangle_img.mean() + 10, rectangle_img.mean() * 2, None, 3), 'ab', 1)
    angles_range = 30
    left_side = rectangle_img[radius_start:, 180 - 10:180 + angles_range]
    right_side = rectangle_img[radius_start:, 360 - angles_range:]  # -30:0
    # h_radius, h_radius2 = right_side.shape[0], right_side.shape[0] // 2
    # mean, std = [], []
    # for rect in [left_side, right_side]:
    #     sob = sobel_h(rect).T  # we look for edges in horizontal direction so we apply sobel_h
    #     edge_points = []
    #     # line detection: we find two peaks, one on side where < radius/2 and another on...
    #     for i, s in enumerate(sob):
    #         if s.max() == 0:
    #             continue
    #         edge_inside_iris, edge_out_iris = s[:h_radius2].argmax(), s[h_radius2:].argmax() + h_radius2  # get max on left and on right side
    #         maximas = signal.argrelmax(s, order=3)[0]  # get local maximas
    #         if edge_inside_iris in maximas and edge_out_iris in maximas:
    #             inside_iris, out_iris = 0, h_radius - 1
    #             between_edges = (edge_inside_iris + edge_out_iris) // 2
    #             edge_points.append([rect[inside_iris, i], rect[edge_inside_iris, i], rect[between_edges, i],
    #                                 rect[edge_out_iris, i], rect[out_iris, i]])
    #         # else:
    #         #     [plt.axvline(x=m, color='r', linestyle='dashed', linewidth=2) for m in maximas]
    #         #     plt.plot(s)
    #         #     plt.axvline(x=edge_inside_iris, color='b', linestyle='dashed', linewidth=2)
    #         #     plt.axvline(x=edge_out_iris, color='b', linestyle='dashed', linewidth=2)
    #         #     plt.draw()
    #         #     plt.waitforbuttonpress(0)  # this will wait for indefinite time
    #         #     plt.close()
    #     std.append(np.array(edge_points).std(axis=0))
    #     mean.append(np.array(edge_points).mean(axis=0))
    # mean_left, mean_right = mean[0], mean[1]
    # std_left, std_right = std[0], std[1]
    # values [inside_iris, edge_inside_iris, between two edges, edge_out_iris, out_iris]
    thresh, th2 = cv2.threshold(left_side, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    std1, std2 = np.std(right_side), np.std(left_side)
    minstd = min(std1, std2)
    m1, m2 = np.mean(right_side), np.mean(left_side)
    minmean = min(m1, m2)
    # if isarcus is not None:
    #     if not np.isnan(mean_left).any() or not np.isnan(mean_right).any():
    #         if np.isnan(mean_right).any():
    #             mean_right = mean_left
    #             std_right = std_left
    #         if np.isnan(mean_left).any():
    #             mean_left = mean_right
    #             std_left = std_right
    #         columns = {'isarcus': [isarcus], 'inside_iris_l': [mean_left[0]],  'edge_inside_iris_l': [mean_left[1]],
    #                    'between_edges_l': [mean_left[2]], 'edge_out_iris_l': [mean_left[3]], 'out_iris_l': [mean_left[4]],
    #                    'inside_iris_r': [mean_right[0]],  'edge_inside_iris_r': [mean_right[1]], 'between_edges_r': [mean_right[2]],
    #                    'edge_out_iris_r': [mean_right[3]], 'out_iris_r': [mean_right[4]],
    #                    'std_inside_iris_l': [std_left[0]],  'std_edge_inside_iris_l': [std_left[1]],
    #                    'std_between_edges_l': [std_left[2]], 'std_edge_out_iris_l': [std_left[3]], 'std_out_iris_l': [std_left[4]],
    #                    'std_inside_iris_r': [std_right[0]],  'std_edge_inside_iris_r': [std_right[1]], 'std_between_edges_r': [std_right[2]],
    #                    'std_edge_out_iris_r': [std_right[3]], 'std_out_iris_r': [std_right[4]],
    #                    'thresh': [thresh], 'minstd': minstd, 'minmean': minmean}
    #         df = pd.DataFrame(columns)
    #         df = df.set_index('isarcus')
    #         df.to_csv('arcus.csv', mode='a', header=False)
    return thresh > 100 and minstd > 28 or minstd > 26 or minmean > 100


def get_contour_compactness(boundary):
    boundary = boundary.astype(np.int32)
    perimeter = cv2.arcLength(boundary, True)
    area = cv2.contourArea(boundary)
    return get_compactness(area, perimeter) if area > 10 else 1000


def get_compactness(area, perimeter):
    return (4 * np.pi * area) / perimeter**2


def get_darker_center(img, c1, c2):
    if c2 is None:
        return c1
    if c1 is None:
        return c2
    v1 = img[c1[1], c1[0]]
    v2 = img[c2[1], c2[0]]
    return c1 if v1 < v2 else c2


def is_iris_diff_arcus(radius_iris_out, radius_iris_in, coeff=0.27):
    return radius_iris_out - radius_iris_in < radius_iris_out * coeff


def ismiosis(radius_pupil, radius_iris):  # small pupil
    return radius_pupil * 3.2 < radius_iris


def ismydriasis(radius_pupil, radius_iris):  # big pupil
    return radius_pupil * 2 > radius_iris


def is_mydriasis_miosis(radius_pupil, radius_iris):
    return ismydriasis(radius_pupil, radius_iris) or ismiosis(radius_pupil, radius_iris)


def circle_hist_similiarity(img_no_reflection, center_iris_in, radius_iris_in, center_pupil, radius_pupil, hist_method=cv2.HISTCMP_INTERSECT):
    """
    Params:
    -------
    hist_method: cv2.HISTCMP_CORREL, cv2.HISTCMP_INTERSECT, cv2.HISTCMP_KL_DIV, cv2.HISTCMP_BHATTACHARYYA
    """
    if center_pupil is not None and center_iris_in is not None:
        c1 = get_circle_pixels(img_no_reflection, center_iris_in, radius_iris_in, 1)
        c2 = get_circle_pixels(img_no_reflection, center_pupil, radius_pupil, 1)
        symmetric_difference = img_no_reflection[np.where(abs(c1 - c2))]
        intersection = img_no_reflection[np.where(abs(c1 * c2))]
        bins = np.linspace(0, 255, 25)
        hist, bin_edges = np.histogram(symmetric_difference, bins=bins)
        hist2, bin_edges2 = np.histogram(intersection, bins=bins)
        diff = cv2.compareHist(hist.astype('float32'), hist2.astype('float32'), hist_method) / np.min((hist2.sum(), hist.sum()))
        plt.bar(bin_edges[:-1], hist, width=10, alpha=0.5, label='outside') and plt.xlim(min(bin_edges), max(bin_edges))
        plt.bar(bin_edges[:-1], hist2, width=10, alpha=0.5, label='intersection') and plt.xlim(min(bin_edges), max(bin_edges))
        plt.legend()
        t = str(symmetric_difference.mean() - intersection.mean()) + '  ,  ' + str(np.argmax(hist) - np.argmax(hist2)) + \
            '  ,  ' + str(symmetric_difference.mean()) + '  ,  ' + str(intersection.mean())
        plt.title(t)
        plt.show()
        return symmetric_difference.mean() - intersection.mean() > 40
    return False


class Timer(ContextDecorator):
    """A better timed class. This uses the ContextDecorator, which allows us
    to use this as a decorator, too!
    """

    def __enter__(self):
        self.start = time()
        print("Starting at {}".format(self.start))
        return self

    def __exit__(self, type, value, traceback):
        self.end = time()
        total = self.end - self.start
        print("Ending at {} (total: {})".format(self.end, total))


def detect_diseases_in_pupil(img, imgWithReflection, center_pupil, radius_pupil, center_iris_out, radius_iris_out, center_iris_in, radius_iris_in, diseases):
    problems = []
    return set()
    diff_dist = euclidean_distance(center_pupil, center_iris_out)
    print('diff_dist:' + str(diff_dist))
    if is_mydriasis_miosis(radius_pupil, radius_iris_out) or diff_dist > 10:
        # get new pupil
        cv = _chan_vese.chan_vese(img, mu=0.75, lambda1=80, lambda2=4, tol=1e-3, max_iter=100,
                                  dt=0.5, init_level_set=cv_disk(img.shape, get_darker_center(img, center_iris_out, center_iris_in), 10), extended_output=True)
        # cv = _chan_vese.chan_vese(img, mu=0.35, lambda1=100, lambda2=2, tol=1e-3, max_iter=100,
        #                           dt=0.5, init_level_set=cv_disk(img.shape, center_iris_out, 10), extended_output=True)
        # cv = _chan_vese.chan_vese(img, mu=0.25, lambda1=100, lambda2=2, tol=0.05, max_iter=500,
        #                           dt=0.5, init_level_set=cv_disk(img.shape, center_iris_out, 10), extended_output=True)
        # plotChanVese(cv, img)
        temp = np.zeros(np.shape(cv[0]), dtype=np.uint8)
        temp[np.where(cv[0])] = 1
        # imshow(np.multiply(img, temp), 'chan vese')
        # res = skimage.measure.regionprops(temp)
        # reg = max(res, key=lambda a: a.area)
        # center_pupil, radius_pupil = (reg.centroid[1], reg.centroid[0]), radiusBBox(reg.bbox)
        a, contours, hier = cv2.findContours(temp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boundary = max(contours, key=lambda a: len(a))
        area = cv2.contourArea(boundary)
        if is_cataract(imgWithReflection, np.where(cv[0])):
            diseases.append(EyeDiseases.Cataract)
        if area < np.pi * radius_pupil**2:
            return (diseases, problems), center_pupil, radius_pupil

        (RMSE, surface, center_pupil, radius_pupil) = get_contour_boundary(boundary)
        if radius_pupil * 1.5 > radius_iris_out and center_iris_in is None or area * 5 > np.pi * radius_iris_out**2 and center_iris_in is None:
            center_pupil, radius_pupil = center_iris_out, radius_iris_out
            # diseases.append(EyeDiseases.Cataract)
            return (diseases, problems), center_pupil, radius_pupil
    # elif is_cataract_from_radius(imgWithReflection, center_pupil, radius_pupil):
    #     diseases.append(EyeDiseases.Cataract)

    return (diseases, problems), center_pupil, radius_pupil


def chan_vese_pupil(img, center_iris_in, center_iris_out, lambda2=7):
    cv = _chan_vese.chan_vese(img, mu=0.75, lambda1=100, lambda2=lambda2, tol=1e-1, max_iter=60,
                              dt=0.5, init_level_set=cv_disk(img.shape, get_darker_center(img, center_iris_out, center_iris_in), 10), extended_output=True)
    # plotChanVese(cv, img)

    temp = np.zeros(np.shape(cv[0]), dtype=np.uint8)
    temp[np.where(cv[0])] = 1
    a, contours, hier = cv2.findContours(temp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if contours == []:
        return None, 0
    boundary = max(contours, key=lambda a: len(a))
    (__, surface, center, radius) = get_contour_boundary(boundary)
    # drawCircle(img, center, radius)
    return center, radius


def is_circle_outside(center_pupil, radius_pupil, center_iris_out, radius_iris_out, safe_factor=1.1):
    """We calculate distance between centers and then we add radius_pupil. We get new radius which is pupil radius + distance between centers.
        If new radius is > radius iris then pupil is outside iris."""
    return euclidean_distance(center_pupil, center_iris_out) + radius_pupil * safe_factor > radius_iris_out if center_iris_out and center_pupil else True


def is_circle_inside(center_pupil, radius_pupil, center_iris_out, radius_iris_out, safe_factor=1.1):
    return not is_circle_outside(center_pupil, radius_pupil, center_iris_out, radius_iris_out, safe_factor)


def detect_darkness(img_hsv, center_iris_in, radius_iris_in):
    "Here we are looking for dark color in hsv space. We measure darkness of iris."
    # x, y, ar = st.locate_corneal_reflection(img_gray, center_iris_out[0], center_iris_out[1], radius_iris_out)
    radius_iris_in = int(radius_iris_in * 0.9)
    if center_iris_in:
        cropped, _ = crop_image(img_hsv, center_iris_in, radius_iris_in)
        # maskb = cv2.inRange(cropped, (0, 0, 0), (180, 125, 80))  # something between
        # maskb = cv2.inRange(cropped, (0, 205, 0), (180, 255, 80))  # on the edge (fail when eye is dark brown)
        mask1 = cv2.inRange(cropped, (0, 0, 0), (180, 255, 50))  # look for dark color
        mask2 = cv2.inRange(cropped, (0, 0, 225), (180, 20, 255))  # very bright
        mask = mask1 + mask2
        # img_gray = cv2.cvtColor(hsv2rgb(cropped), cv2.COLOR_BGR2GRAY)
        # img_gray[(mask1 + mask2).astype(bool)] = 255
        # imshow3(img_gray, '', 1)
        # img_gray = cv2.cvtColor(hsv2rgb(cropped), cv2.COLOR_BGR2GRAY)
        # img_gray[(maskb).astype(bool)] = 255
        # imshow3(img_gray, '', 1)
        # imshow3(hsv2rgb(img_hsv),'', 1)
        return mask.sum() / 2.55 / mask.size


def laplacian_edge_test(image):
    lap = cv2.Laplacian(image, cv2.CV_64F)
    lap = cv2.convertScaleAbs(lap)
    for i in range(10, 3, -1):
        lap[lap > i] = 255
        imshow3(lap)


def is_iris_pupil(img_color, img_hsv, center_iris, radius_iris):
    """If iris is pupil return True otherwise False.

    Note:
    ----
    False is returned if image is gray or center_iris is none or image contains more than 25% of brown color.
    True is returned if image is dark."""
    if center_iris is None or isimagegray(img_color) or detect_brown_color(img_hsv, center_iris, radius_iris) > 25:
        return False  # iris is not pupil if we detect more than 25% of brown color
    if detect_darkness(img_hsv, center_iris, radius_iris) > 70:
        return True  # 75% is the are of circle in square - some noise
    colorfulness = image_colorfulness(img_color, center_iris, radius_iris)
    return colorfulness < 9


def detect_brown_color(img_hsv, center_iris_out, radius_iris_out):
    "Measure of brown color in HSV space."
    cropped = img_hsv[center_iris_out[1] - radius_iris_out // 2:center_iris_out[1] +
                      radius_iris_out // 2, center_iris_out[0] - radius_iris_out:center_iris_out[0] + radius_iris_out]
    min_brown = cv2.inRange(cropped, (0, 100, 20), (25, 255, 255))
    # mask1 = cv2.inRange(cropped, (0, 100, 20), (10, 255, 255))
    # mask2 = cv2.inRange(cropped, (170, 100, 20), (180, 255, 255))
    bw = min_brown
    # bw = mask2 | mask1
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # bw = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # morfological opening! remove small regions
    # imshow(bw, 'brown color')
    percent = bw.sum() / 2.55 / bw.size
    return percent


def glcm_method(croped):
    from sklearn.metrics.cluster import entropy
    from skimage.feature import greycomatrix, greycoprops
    glcm = greycomatrix(croped, distances=[1], angles=[np.pi / 4, 3 * np.pi / 4], symmetric=True, normed=True)
    feats = {prop: np.sum(greycoprops(glcm, prop).ravel()) for prop in ['ASM', 'energy']}
    print(feats)


def image_colorfulness(img_color, center_iris, radius_iris):
    indices = cv2.circle(np.zeros((img_color.shape[:2])), center_iris, roundToInt(radius_iris), 1, -1).astype(np.bool)
    image = img_color[indices].astype("float")
    # split the image into its respective RGB components
    (B, G, R) = cv2.split(image) if len(image.shape) == 3 else (image[:, 0], image[:, 1], image[:, 2])
    # compute rg = R - G
    rg = np.absolute(R - G)

    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)

    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))

    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)


def is_hsv_gray_from_radius(img_hsv, center, radius):
    temp = np.zeros_like(img_hsv[:, :, 1])
    temp = cv2.circle(temp, tuple(roundToInt(center)), roundToInt(radius), (255, 255, 255), -1)
    return is_hsv_gray(img_hsv, np.where(temp))


def is_hsv_gray(img_hsv, indices):
    s = np.mean(img_hsv[:, :, 1][indices])
    v = np.mean(img_hsv[:, :, 2][indices])
    print('saturation: %s, value: %s' % (str(s), str(v)))
    return s < saturation_gray_limit()


def saturation_gray_limit():
    return 85


def remove_small_areas(bw):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)  # morfological opening! remove small regions
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)  # morfological closing! fill region
    return bw

    # from skimage.restoration import inpaint


def corneal_and_inpating():
    # x, y, ar = st.locate_corneal_reflection(img_gray, center_iris_out[0], center_iris_out[1], radius_iris_out)
    # mask = np.zeros_like(img_gray)
    # mask[y - ar * 2:y + ar * 2, x - ar * 2:x + ar * 2] = 1
    # img_gray = cv2.inpaint(img_gray, mask, 3, cv2.INPAINT_NS)
    pass


def closest_circles_radius_distance(center1, radius1, center2, radius2):
    dist = euclidean_distance(center1, center2)
    is_circle_outside()
    euclidean_distance(center1, center2) + radius_pupil * safe_factor >= radius_iris_out if center1 and center2 else False


def isimagegray(image):
    (B, G, R) = cv2.split(image)
    return (B - G).sum() == 0 and (G - R).sum() == 0


def crop_image(img, center, radius):
    """Returns cropped image with top left corner (y,x). This is if iris radius is out of image.
    """
    x, y = center
    topy = y - radius if radius < y else 0
    leftx = x - radius if radius < x else 0
    if img.ndim == 3:
        return img[topy:y + radius, leftx:x + radius, :], np.array([topy, leftx])
    return img[topy:y + radius, leftx:x + radius], np.array([topy, leftx])  # go over high limit is not a problem


def showLuvChannel(img_color):
    temp = cv2.cvtColor(img_color, cv2.COLOR_BGR2Luv)
    cv2.imshow('1', cv2.equalizeHist(temp[:, :, 0]))
    cv2.imshow('2', cv2.equalizeHist(temp[:, :, 1]))
    cv2.imshow('3', cv2.equalizeHist(temp[:, :, 2]))
    c = cv2.equalizeHist(temp[:, :, 2]).astype(float) + cv2.equalizeHist(temp[:, :, 1]).astype(float)
    imshow3(c / np.max(c), 'combined', 1)


def detect_diseases_in_pupil_snake(img_color, img_hsv, img_no_reflection, img_gray, center_pupil, radius_pupil, center_iris_out,
                                   radius_iris_out, center_iris_in, radius_iris_in, compactness, diseases, verbose=0):
    "First k-means algoritm is applied on gray image. "
    # pupilmean = img_no_reflection[get_circle_pixels(img_no_reflection, center_pupil, radius_pupil, 1).astype(bool)].mean()
    # # get points that are in pupil_convexhull and not in pupil (points around pupil circle)
    # points_notin_pupil = get_circle_pixels(img_gray, center_pupil, radius_pupil, 1) - pupil_convexhull
    # points_notin_pupil_mean = img_no_reflection[points_notin_pupil.astype(bool)].mean()
    # if compactness < 2 and pupilmean + 10 < points_notin_pupil_mean:
    #     pupilmean1 = img_gray[get_circle_pixels(img_no_reflection, center_iris_out, radius_pupil, 1).astype(bool)].mean()  # check if iris is on right place
    #     if not is_circle_outside(center_pupil, radius_pupil, center_iris_out, radius_iris_out) or pupilmean1 > 50:
    #         return diseases, center_pupil, radius_pupil, None, None  # if not big diff beetwen kmeans region and kasa method then is all normal
    # if False and euclidean_distance_(pupil_coords, center_iris_out).mean() < radius_iris_out * 0.7 \
    #         and any(points_in_poly([center_iris_out, (0, 0) if center_iris_in is None else center_iris_in], np.squeeze(pupil_coords))) \
    #         and check_normal_pupil(radius_pupil, radius_pupil1, radius_iris_out, radius_iris_in, center_pupil, center_pupil1, center_iris_out):
    #     # if mean (euclidian distance (center_iris_out, pupil coords)) < radius_iris_out * 0.7 "if mean of pupil points is not far away from center"
    #     # and if any of center_iris_out, center_iris_in in coords then all normal and return
    #     return diseases + diseases1, center_pupil1, radius_pupil1, pupil_coords, pupil_convexhull
    if True:
        # pupilmean1 = img_gray[get_circle_pixels(img_no_reflection, center_iris_out, radius_pupil, 1).astype(bool)].mean()  # check if iris is on right place
        # diff_dist = min((euclidean_distance(center_pupil, center_iris_out), euclidean_distance(center_pupil, center_iris_in)))
        # if diff_dist < 15 and compactness < 1.3 or pupilmean1 > 90:
        #     return diseases, center_pupil, radius_pupil, None, None  # all normal
        # if center_iris_in:
        #     center_iris_out, radius_iris_out = center_iris_in, radius_iris_in
        # import houghTransform as ht
        # radius, center = ht.detect_circle_pupil(test, roundToInt(np.array(test.shape) / 2))
        # drawCircle(test, center, radius)
        # center = np.array(center) + np.array(center_iris_out) - int(radius_iris_out * 0.7)
        # def callback(snake):
        #     t = img_no_reflection.copy()
        #     t[snake == 1] = 255
        #     imshow3(t.astype(np.uint8), '', 1)

        # snake = morphological_chan_vese(img_no_reflection, 50, init_level_set=circle_level_set(img_gray.shape, (center_iris_out[1], center_iris_out[0]), 10),
        #                                 lambda1=75, iter_callback=callback)
        # s = np.linspace(0, 2 * np.pi, 400)
        # x = center_iris_out[0] + radius_iris_out * 0.8 * np.cos(s)
        # y = center_iris_out[1] + radius_iris_out * 0.8 * np.sin(s)
        # init = np.array([x, y]).T
        # from skimage.filters import gaussian
        # t = cv2.medianBlur(img_color, 15)
        # imshow3(t, '', 1)
        # snake = active_contour(t / 255., init, w_edge=5, gamma=2, alpha=2, beta=0.9)
        # fig, ax = plt.subplots(figsize=(7, 7))
        # ax.imshow(img_gray, cmap=plt.cm.gray)
        # ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
        # ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
        # ax.set_xticks([]), ax.set_yticks([])
        # ax.axis([0, img_gray.shape[1], img_gray.shape[0], 0])
        # plt.show()
        # imshow3((snake * 255).astype(np.uint8), '', 1)
        # reg = regionprops(snake)[0]
        # center, radius = np.array(reg.centroid)[[1, 0]], reg.equivalent_diameter / 2(
        # for i in range(0, 8):
        #     # Display the image and plot all contours found
        #     fig, ax = plt.subplots()
        #     ax.imshow(regions, interpolation='nearest', cmap=plt.cm.gray)
        #     for n, contour in enumerate(contours):
        #         ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        #     ax.axis('image')
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #     plt.show()
        # reg = regionprops(regions)
        # reg = max(reg, key=lambda a: a.area)  # get darkest region, this is with max area
        # import bresenham_circle as bc
        # bc.test(cropped)
        # diseases1, center_pupil1, radius_pupil1, pupil_coords, pupil_convexhull, meancolor = pupil_with_kmeans(
        #     img_color, img_gray, center_iris_out, radius_iris_out, diseases, verbose=1)
        # drawCircle(img_gray, center_pupil1, radius_pupil1)
        # plot_snake(img_no_reflection, init, roundToInt(snake))
        # while radius < 7 or radius > snake_radius:
        # center, radius = chan_vese_pupil(img_no_reflection, center_iris_in, center_iris_out, lambda2=6)
        #     break
        # c, r = center_iris_out, radius_iris_out
        # face = img_gray[c[1] - r:c[1] + r, c[0] - r:c[0] + r]
        # compactness1 = get_contour_compactness(snake)
        # if compactness1 > 3:
        # center, radius = chan_vese_pupil((c * 255).astype(np.uint8), center_iris_in, center_iris_out)
        # what fits better to iris
        # if diff_dist < min((euclidean_distance(center, center_iris_out), euclidean_distance(center, center_iris_in))) * 1.5:
        return diseases, center, radius, pupil_coords, pupil_convexhull


def whitebalance(img_color):
    cv2.imshow('a', img_color)
    img_color1 = img_color.copy()
    cv2.xphoto.balanceWhite(img_color, img_color1, 0)
    cv2.imshow('b', img_color1)
    cv2.waitKey(0)
    return img_color1


def pupil_with_kmeans(img_color, img_gray, center_iris_out, radius_iris_out, diseases, verbose=0):
    iris_mask = get_circle_pixels(img_gray, center_iris_out, radius_iris_out, max_value=1)
    regs_kmeans, labels = segmentation.kmeans(img_gray, K=3)
    regs = regs_kmeans.copy()
    regs[regs != labels.pop(0)] = 255
    inside_iris = remove_small_areas(invert(regs) * iris_mask)
    inside_iris[inside_iris == 0] = inside_iris.max() + 1
    reg = regionprops(label(inside_iris))
    reg = max(reg, key=lambda a: a.area)  # get darkest region, this is with max area
    # imshow3(regs, 'regs_kmeans', verbose=verbose)
    # imshow3(regs_kmeans, verbose=verbose)
    topleft = np.array(reg.bbox[:2])  # save top left coordinates for later
    if is_cataract_calculate(img_gray, tuple(reg.coords.transpose()), verbose):  # if it is cataract on kmeans segmentation
        diseases.append(EyeDiseases.Cataract)
    convex = reg.convex_image.astype(np.uint8)
    a, contours, hier = cv2.findContours(convex.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    coords = contours[0] + topleft[[1, 0]]  # countur is in local coordinate system
    reg = regionprops(convex)[0]
    # compactness = get_contour_compactness(contours[0])
    # if compactness > 1.237:
    #     diseases.append(EyeDiseases.IritisOrKeratitisPupilShape)
    # print('ec:' + str(reg.eccentricity))
    # print('compactness: ' + str(compactness))
    return diseases, (np.array(reg.centroid) + topleft)[[1, 0]], reg.equivalent_diameter / 2, coords, convex


def aspect_ratio(bbox):
    x, y, w, h = bbox
    if h > w:
        w, h = h, w
    return abs(float(w) / h)


def normal_distributed():
    pass


def euclidean_distance_(pupil_coords, center_iris_out):
    return np.sum(np.subtract(np.squeeze(pupil_coords), center_iris_out)**2, axis=1)**0.5


def gaussian_mixture_model_segmentation(img_hsv, img_no_reflection, center_iris_out, radius_iris_out, k=8, force=True, verbose=0):
    cropped1, _ = crop_image(img_no_reflection, center_iris_out, radius_iris_out)
    cropped, topleft = crop_image(img_hsv, center_iris_out, radius_iris_out)
    center_local = np.array(cropped1.shape) / 2
    regions = segmentation.gaussianmixturemodel(cropped, k, verbose)
    contours = label(regions)
    # imshow3(contours / contours.max(), 'sdas', 1)
    regions = regionprops(contours, intensity_image=cropped1)
    regions = list(filter(lambda r: r.area > np.pi * 25, regions))
    if regions == []:
        return center_iris_out, radius_iris_out, None, None, 0
    # we look for radius close to radius_iris/2, but also punish higher distance from center
    # everything multipled (how much pixels are in bbox area)/allpixels (11) * mean_intensity of pixels
    # we take radius_iris_out/2 - radius_pupil == 0 -> + region_dist from center * (r.bbox_area / r.area)  * r.mean_intensity
    regions = sorted(regions, key=lambda r: (abs(radius_iris_out / 2.2 - radius_bbox(r.bbox)) +
                                             euclidean_distance((r.centroid[1], r.centroid[0]), center_local))
                     * (r.bbox_area / r.area) * (r.mean_intensity + 1)**1.5 * aspect_ratio(r.bbox))  # get darkest region, this is with max area
    # regions = sorted(regions, key=lambda r: (abs(radius_iris_out / 2 - radius_bbox(r.bbox))
    #                                          + euclidean_distance((r.centroid[1], r.centroid[0]), center_local)
    #                                          + ((r.mean_intensity + 1) / 5)
    #                                          )
    #                  * (r.bbox_area / r.area)**2
    #                  )  # get darkest region, this is with max area
    bestreg = regions[0]
    for j, r in enumerate(regions):
        if (regions[j].mean_intensity < 9 and regions[j].area > bestreg.area) \
                and regions[j].bbox_area > bestreg.bbox_area \
                and euclidean_distance_(regions[j].coords, center_local).max() < cropped1.shape[0] / 2 * .7:
            bestreg = r
        if j > 1:
            continue
        # print('*' * 50, r.eccentricity, r.area, r.bbox_area)
        # print('centroid: ', (r.centroid[1], r.centroid[0]), center_local)
        # # print('radius: ', radius_bbox(r.bbox), radius_iris_out)
        # print('aspect_ratio: ', aspect_ratio(r.bbox))
        # print('radius away from expected: ', abs(radius_iris_out / 2.2 - radius_bbox(r.bbox)))
        # print('centroid diff: ', euclidean_distance((r.centroid[1], r.centroid[0]), center_local))
        # print('mean_intensity: ', r.mean_intensity)
        if verbose > 0:
            print('Gmm score for region:', (abs(radius_iris_out / 2.2 - radius_bbox(r.bbox)) +
                                            euclidean_distance((r.centroid[1], r.centroid[0]), center_local))
                  * (r.bbox_area / r.area) * (r.mean_intensity + 1)**1.5 * aspect_ratio(r.bbox))
        # res = MarkImage(cropped1, r.coords[:, 0], r.coords[:, 1])
        # imshow3(res, 'gaussian mixture module', 1)
    coords = bestreg.coords + topleft
    l = regionprops(bestreg.convex_image.astype(int))[0]
    convex = np.transpose(np.where(bestreg.convex_image)) + topleft + bestreg.bbox[:2]
    coords, convex = tuple(coords.transpose()), tuple(convex.transpose())  # prepare for array indexing
    center, radius = (np.array(bestreg.centroid) + topleft)[[1, 0]], l.equivalent_diameter // 2  # radius_bbox(bestreg.bbox)
    a = img_no_reflection.copy()
    a[coords] = 255
    imshow3(a, '1', verbose)
    return tuple(roundToInt(center)), radius, coords, convex, get_compactness(l.area, l.perimeter)


def probabilites(radius_iris_out):
    from scipy.stats import norm
    x = np.linspace(0, radius_iris_out)
    plt.plot(x - radius_iris_out / 2, norm.pdf(x, radius_iris_out / 2, radius_iris_out / 2), 'r-', lw=5, alpha=0.6, label='norm pdf')
    plt.show()

    from scipy.stat import triang
    x = np.linspace(0, radius_iris_out)
    c = 1
    plt.plot(x, triang.pdf(x, 0.5, scale=c * 2), 'r-', lw=5, alpha=0.6, label='triang pdf')
    plt.show()
    triang.pdf(radius_iris_out, 0.5, scale=radius_iris_out)


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values - average)**2, weights=weights)  # Fast and numerically precise
    return (average, np.sqrt(variance))


import houghTransform as ht


def statwithhis():
    pass
    # from scipy.stats import norm
    # from sklearn.neighbors import KernelDensity
    # kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(samp)
    # X_plot = np.linspace(0, 255, 25)
    # # log_dens = kde.score_samples(X_plot)
    # plt.hist(samp, 25, [0, 256])
    # hist, bin_edges = np.histogram(samp, X_plot)
    # weights = [hist[i - 1] for i in np.digitize(samp, bin_edges)]
    # plt.title(np.round([*ut.weighted_avg_and_std(samp, weights), samp.mean(), np.std(samp)]))
    # plt.show()
    # plt.plot(np.linspace(0, 255, 25), np.exp(log_dens))
    # param = stats.rayleigh.fit(samp)  # distribution fitting
    # pdf_fitted = stats.rayleigh.pdf(x, loc=param[0], scale=param[1])
    # param = stats.expon.fit(samp)
    # pdf1 = stats.expon.pdf(x, loc=param[0], scale=param[1])
    # plt.title(np.std(samp))
    # plt.plot(x, pdf_fitted, 'r-', x, pdf1, 'b-')
    # plt.hist(samp, normed=1, alpha=.3)
    # plt.legend()
    # plt.show()


def get_vector_of_diseases(sublist) -> list:
    """Return list of diseases. If input is no diseases, one disease or more.
    In all cases list is created filled with healty values. This is for confussion matrix

    e.g. from disease [miosis ]we get vector for confusion matrix. e.g.[healthy eye, miosis, healthy eye,...]"""
    mylist = [EyeDiseases.HealtyEye.value] * len(EyeDiseases.diseases_table())
    for item in sublist:
        ind = EyeDiseases.diseases_table().index(item.value)
        mylist[ind] = item.value
    return mylist


def find_new_iris(img_equalized, img_medianblured, radius_iris_out, center_iris_out, center_pupil, radius_pupil,
                  center_iris_in, radius_iris_in, canny_thresh, canny_factor):
    """Return new iris.

    Note:
    ----
    step: min distance from pupil -> radius_pupil*step
    """
    step = 1
    # if we find new iris but is crossing pupil then we increase step
    while is_circle_outside(center_pupil, radius_pupil, center_iris_out, radius_iris_out):  # if iris is too close to pupil find new iris
        step += 0.1
        radius_iris_in1, center_iris_in1, radius_iris_out1, center_iris_out1, canny_thresh = ht.detect_circle_iris(
            img_medianblured, center_pupil, radius_iris_out * step, canny_factor, canny_low_thresh=canny_thresh)
        if not is_circle_outside(center_pupil, radius_pupil, center_iris_out1, radius_iris_out1):
            # save good result, this is in case if we cannot find new better iris then same iris is returned
            radius_iris_in, center_iris_in, radius_iris_out, center_iris_out = radius_iris_in1, center_iris_in1, radius_iris_out1, center_iris_out1
        if step >= 1.3:
            break
    if is_circle_outside(center_pupil, radius_pupil, center_iris_out, radius_iris_out):  # we have to find iris
        _, _, radius_iris_out, center_iris_out, _ = ht.detect_circle_iris(img_equalized, center_pupil, radius_iris_out * 1.5, canny_factor)
    return radius_iris_in, center_iris_in, radius_iris_out, center_iris_out


def plot_snake(img, init, snake):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    plt.gray()
    ax.imshow(img)
    ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
    ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])
    plt.show()
    # , centerIris
    # if centeriris is not None:
    #     _analyzeCircle(img, centeriris, radius_iris)


def is_iritis_or_keratitis(img_gray, center_pupil, radius_pupil, convexhull, verbose):
    if convexhull is not None:
        pupil_region = get_circle_pixels(img_gray, center_pupil, radius_pupil, max_value=1)
        # res = MarkImage(img_gray, pupil_convexhull[:, 0], pupil_convexhull[:, 1])
        # imshow3(res, '1', 1)
        convex_region = np.zeros_like(img_gray)
        convex_region[convexhull] = 1
        res = regionprops(convex_region)
        reg = max(res, key=lambda a: a.area)
        intersection = (pupil_region * convex_region).sum()
        union = (pupil_region + convex_region).astype(bool).sum()
        jaccard_index = intersection / union
        compactness = get_compactness(reg.area, reg.perimeter)
        if verbose > 0:
            # print('diff between circle and kmeans clustering convexhull:', jaccard_index)
            print('is_iritis_or_keratitis Roundness: ', compactness)
        return compactness < .87, reg.equivalent_diameter // 2 if compactness < 0.6 else radius_pupil
        # if jaccard_index < 0.75:
        #     return True
    return False, 0


def detect_diseases_in_iris(img_gray, img_medianblured, center_pupil, radius_pupil, center_iris_out, radius_iris_out, center_iris_in, radius_iris_in,
                            pupil_convexhull, verbose):
    diseases = []
    # return set([])
    if EyeDiseases.exist('ArcusSenilis') and is_arcus_senilis(img_medianblured, center_pupil, radius_pupil, center_iris_out, radius_iris_out, center_iris_in, radius_iris_in):
        diseases.append(EyeDiseases.ArcusSenilis)
    if ismiosis(radius_pupil, radius_iris_out):
        diseases.append(EyeDiseases.Miosis)
    elif ismydriasis(radius_pupil, radius_iris_out):
        diseases.append(EyeDiseases.Mydriasis)
    temp = is_iritis_or_keratitis(img_gray, center_pupil, radius_pupil, pupil_convexhull, verbose)
    if temp[0]:
        diseases.append(EyeDiseases.IritisOrKeratitisPupilShape)
        if ismiosis(temp[1], radius_iris_out):
            diseases.append(EyeDiseases.Miosis)
    if pupil_convexhull is None and is_cataract_from_radius(img_gray, center_pupil, radius_pupil, verbose):
        diseases.append(EyeDiseases.Cataract)
    elif pupil_convexhull is not None and is_cataract_calculate(img_gray, pupil_convexhull, verbose):
        diseases.append(EyeDiseases.Cataract)
    # if EyeDiseases.Miosis in diseases and EyeDiseases.Cataract in diseases:  # can be false detection if it's cataract
    #     diseases.remove(EyeDiseases.Miosis)
    return set(diseases)


def radius_bbox(bbox):
    x, y, w, h = bbox
    return (abs(w - x) + abs(h - y)) // 4  # /4 is due to: 1. /2 is mean and 2. /2 is perimeter/2 -> radius


def center_bbox(bbox):
    x, y, w, h = bbox
    return x + w // 2, y + h // 2


def draw2Circle(imgg, center, radius, center1, radius1, text=""):
    imgc1 = cv2.cvtColor(imgg, cv2.COLOR_GRAY2BGR)  # enable color drawing
    if center1 is not None:
        imgc1 = cv2.circle(imgc1, tuple(roundToInt(center1)), roundToInt(radius1), (0, 0, 255), 1)
    if center is not None:
        imgc1 = cv2.circle(imgc1, tuple(roundToInt(center)), roundToInt(radius), (0, 255, 0), 1)
    cv2.imshow(text, imgc1)
    return cv2.waitKey(0)


def draw3Circle(imgg, center, radius, center1, radius1, center2, radius2, text="", coords=None, label=None, verbose=1):
    # if coords is None:
    #     return
    # return
    imgc1 = cv2.cvtColor(imgg, cv2.COLOR_GRAY2BGR) if imgg.ndim != 3 else imgg.copy()  # enable color drawing
    if center2 is not None:
        imgc1 = cv2.circle(imgc1, tuple(roundToInt(center2)), roundToInt(radius2), (255, 0, 0), 2)
    if center1 is not None:
        imgc1 = cv2.circle(imgc1, tuple(roundToInt(center1)), roundToInt(radius1), (0, 0, 255), 2)
    if center is not None:
        imgc1 = cv2.circle(imgc1, tuple(roundToInt(center)), roundToInt(radius), (0, 255, 0), 2)
    if coords is not None:
        imgmarked = MarkImage(imgg.copy() if imgg.ndim != 3 else cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY),
                              coords[0], coords[1], -1, (0, 0, 255))
        cv2.putText(imgmarked, 'Gaussian mixture model', (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        imshow3(imgmarked, 'marked', verbose, False)
    if label:
        cv2.putText(imgc1, str(label).replace('{', '').replace('}', '').strip(), (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return imshow3(imgc1, text, verbose)


def drawCircle(imgg, center, radius, text="", verbose=1):
    imgc1 = cv2.cvtColor(imgg, cv2.COLOR_GRAY2BGR)  # enable color drawing
    imgc1 = cv2.circle(imgc1, tuple(roundToInt(center)), roundToInt(radius), (0, 255, 0), 1)
    imshow3(imgc1, text, verbose)


def drawContours(img, contours, text=""):
    imgc = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)  # enable color drawing
    cv2.drawContours(imgc, contours, -1, (0, 255, 0), 1)
    cv2.imshow(text, imgc)
    return cv2.waitKey(0)


def resize_image(img, newWH, WidthOrHeight):
    if WidthOrHeight == 'width':
        if img.shape[1] < newWH:
            return img
    if WidthOrHeight == 'height':
        if img.shape[0] < newWH:
            return img
    imSize = cv_size(img)
    # resize image on new Image Width
    [W, H] = imSize[0:: 1]
    if WidthOrHeight == "width":
        newHeight = np.int(newWH / W * H)
        return cv2.resize(img, (newWH, newHeight), interpolation=cv2.INTER_AREA)
    else:
        newWidth = np.int(newWH / W * H)
        return cv2.resize(img, (newWidth, newWH), interpolation=cv2.INTER_AREA)


def get_circle_simple(boundary):
    xc, yc = np.squeeze(boundary).mean(axis=0)
    r = np.sqrt((boundary[:, 0] - xc)**2 + (boundary[:, 1] - yc)**2).mean()
    return (int(xc), int(yc)), int(r)


def get_contour_boundary(boundary):
    boundary = np.squeeze(boundary)
    M = -(boundary**2).sum(1)
    M = M.reshape((-1, 1))
    N = np.append(boundary, np.ones((np.size(boundary, 0), 1)), axis=1)
    A = np.dot(np.linalg.pinv(N), M)
    y0 = -A[0] / 2
    x0 = -A[1] / 2  # center of circle
    radius = np.nan_to_num(np.sqrt((A[0]**2 + A[1]**2) / 4 - A[2]))  # radius
    # (rmse, surface, center, radius)
    return (np.sqrt(np.mean(np.round(np.dot(N, A) - M)**2)), np.pi * radius**2, (int(np.round(y0)), int(np.round(x0))), int(np.round(radius)))


def getPointsInCircleFast(center, radius):
    xR = list()
    yR = list()
    rSquared = radius**2
    xCenter = center[0]
    yCenter = center[1]
    yMin = yCenter - radius
    xMin = xCenter - radius
    xMax = xCenter + radius
    for x in np.arange(xMin, xCenter) + 1:
        for y in np.arange(yMin, yCenter) + 1:
            if (x - xCenter)**2 + (y - yCenter)**2 <= rSquared:
                xSym = xCenter - (x - xCenter)
                ySym = yCenter - (y - yCenter)
                xR += [x, x, xSym, xSym]
                yR += [y, ySym, y, ySym]


def getPointsInCircle(center, radius, center1, radius1, minMaxShape):
    xR = list()
    yR = list()
    rSquared = radius**2
    xCenter = center[0]
    yCenter = center[1]
    rSquared1 = radius1**2
    xCenter1 = center1[0]
    yCenter1 = center1[1]
    yMin = yCenter - radius
    yMax = yCenter + radius + 1
    xMin = xCenter - radius
    xMax = xCenter + radius + 1
    (H, W) = minMaxShape
    if xMin < 0:
        xMin = 0
    if yMin < 0:
        yMin = 0
    if yMax > H:
        yMax = H
    if xMax > W:
        xMax = W
    for x in np.arange(xMin, xMax):
        for y in np.arange(yMin, yMax):
            if (x - xCenter)**2 + (y - yCenter)**2 <= rSquared:
                if (x - xCenter1)**2 + (y - yCenter1)**2 > rSquared1:
                    xR += [x]
                    yR += [y]
                # (x, y), (x, ySym), (xSym , y), (xSym, ySym) are in the circle
    return (np.asarray(yR), np.asarray(xR))


def getMax(type):
    ii32 = np.iinfo(type)
    return ii32.max


def getPointsBelowParabolaSlow(yC, xC, yParabola, k):
    indices = list()
    for ind, t in enumerate(xC):
        if yParabola[t] + k < yC[ind]:
            indices.append(ind)
    indices = np.asarray(indices)
    return (yC[indices], xC[indices])


def getIndicesBelowParabolaAndAboveLine(yC, xC, a, h, k, yLine):
    y = np.round(a / 1000 * (xC - h)**2 + k).astype(int)
    indices1 = (y < yC)
    indices2 = (yC < yLine)
    sas1 = np.where(indices1)
    sas2 = np.where(indices2)
    return np.intersect1d(sas1, sas2)


def getPointsBelowAboveParabola(yC, xC, a, h, k):
    y = np.round(a / 1000 * (xC - h)**2 + k).astype(int)
    indicesBelow = (y < yC)
    indicesAbove = (y >= yC)
    return ((yC[indicesBelow], xC[indicesBelow]), (yC[indicesAbove], xC[indicesAbove]))


def removePupilFromEdges(edges, center, radius):
    (centerX, centerY) = center
    edges = cv2.circle(edges, center, radius, 0, -1)
    return edges


def im_to_polar(img, m, r_min=0, r_max=1, n=360, radius_start=0):
    """ IMTOPOLAR converts rectangular image to polar form. The output image is
        an MxN image with M points along the r axis and N points along the theta
        axis. The origin of the image is assumed to be at the center of the given
        image. The image is assumed to be grayscale.
        Bilinear interpolation is used to interpolate between points not exactly
        in the image.

        rMin and rMax should be between 0 and 1 and rMin < rMax. r = 0 is the
        center of the image and r = 1 is half the width or height of the image.

        V0.1 7 Dec 2007 (Created), Prakash Manandhar pmanandhar@umassd.edu

        params
        ------
        M: radius
        N: angle
        """
    if img.max() > 1:
        img = img / 255.
    Mr, Nr = img.shape
    # pixels should be transformed from 0-360 in anticlockwise direction
    # for testing how pixels are transformed
    # img[int(Mr / 2 - 20):int(Mr / 2 - 1), Nr - 10:Nr - 1] = 0  # right
    # # imR[0:10, int(Nr / 2 - 5):int(Nr / 2 + 5)] = 0.2  # top
    # img[int(Mr / 2 - 20):int(Mr / 2 - 1), 0:9] = 0.4  # left
    # imR[Mr - 10:Mr, int(Nr / 2 - 5):int(Nr / 2 + 5)] = 1  # bottom
    # size of rectangular image
    Om = (Mr - 1) / 2  # co - ordinates of the center of the image
    On = (Nr - 1) / 2
    # scale factors
    sx = (Mr - 1) / 2
    sy = (Nr - 1) / 2

    imP = np.zeros((m, n))

    delR = (r_max - r_min) / (m - 1)
    delT = 2 * np.pi / n
    angles_range = range(170, n)
    radius_range = range(radius_start, m)
    # loop in radius and
    for ri in radius_range:
        for ti in angles_range:
            r = r_min + ri * delR
            t = ti * delT
            x = r * np.cos(t)
            y = -r * np.sin(t)
            if y < 0:
                pass
            xR = x * sx + Om
            yR = y * sy + On
            if yR < 0:
                pass
            try:
                imP[ri, ti] = interpolate(img, yR, xR)
            except IndexError:
                continue
    return imP


def interpolate(imR, xR, yR):
    xf = int(np.floor(xR))
    xc = int(np.ceil(xR))
    yf = int(np.floor(yR))
    yc = int(np.ceil(yR))
    if xf == xc and yc == yf:
        v = imR[xc, yc]
    elif xf == xc:
        v = imR[xf, yf] + (yR - yf) * (imR[xf, yc] - imR[xf, yf])
    elif yf == yc:
        v = imR[xf, yf] + (xR - xf) * (imR[xc, yf] - imR[xf, yf])
    else:
        A = np.array([[xf, yf, xf * yf, 1],
                      [xf, yc, xf * yc, 1],
                      [xc, yf, xc * yf, 1],
                      [xc, yc, xc * yc, 1]])
        r = np.array([[imR[xf, yf]],
                      [imR[xf, yc]],
                      [imR[xc, yf]],
                      [imR[xc, yc]]])
        a, resid, rank, s = np.linalg.lstsq(A, r)
        w = [xR, yR, xR * yR, 1]
        v = np.dot(w, a)
    return v


# im = np.array([[0, 0, 2, 0, 0, 8, 8],
#                [0, 0, 0, 2, 0, 8, 8],
#                [0, 0, 1, 0, 4, 0, 0],
#                [0, 3, 2, 0, 9, 3, 0],
#                [0, 0, 6, 0, 5, 0, 0],
#                [0, 0, 0, 8, 0, 0, 0],
#                [0, 0, 0, 8, 0, 0, 0]])
# ImToPolar(im, 0, 1, 4)


def removePixelsBasedOnColorValue(image, y, x, DarkestPixelValue=None, BrightestPixelValue=None, DarkestPercentage=None, BrightestPercentage=None):
    numOfPixels = int(y.size)
    imageSelected = image[y, x]
    if DarkestPercentage or DarkestPixelValue:
        sorted = np.sort(imageSelected)
        if DarkestPercentage:
            numOfPixelsToRemove = int(numOfPixels * DarkestPercentage / 100)
            DarkestPixelValue = sorted[numOfPixelsToRemove]
        indices1 = np.where(imageSelected > DarkestPixelValue)
    if BrightestPercentage or BrightestPixelValue:
        sorted = np.sort(imageSelected)
        if BrightestPercentage:
            numOfPixelsToRemove = int(numOfPixels * BrightestPercentage / 100)
            BrightestPixelValue = sorted[-numOfPixelsToRemove]
        indices2 = np.where(imageSelected < BrightestPixelValue)
    if BrightestPercentage is None and BrightestPixelValue is None:
        indices = indices1[0]
    elif DarkestPercentage is None and DarkestPixelValue is None:
        indices = indices2[0]
    elif not (BrightestPercentage is None and BrightestPixelValue is None and DarkestPercentage is None and DarkestPixelValue is None):
        indices = np.intersect1d(indices1[0], indices2[0])
    y = y[indices]
    x = x[indices]
    return (y, x)


def MarkImage(image, y, x, text="marked", color=(255, 0, 0)):
    if(image.ndim != 3):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # enable color drawing
    image[y, x] = color
    return image


def DrawMarkImage(image, y, x, text="marked"):
    if(image.ndim != 3):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # enable color drawing
    image[y, x] = [255, 0, 0]
    cv2.imshow(text, image)
    cv2.waitKey(0)


def getLowerHalfOfCircle(y, x, center):
    indices = np.where(y > center[1])
    return (y[indices], x[indices])


def plotPoints(imgg, x, y, vertex=None):
    if(vertex == None):
        vertex = (0, 0)
    implot = plt.imshow(imgg, cmap='gray')
    plt.plot(x, y, color='blue')
    plt.plot(vertex[0], vertex[0], color='red')
    plt.show()


def plotChanVese(cv, imgg):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    ax = axes.flatten()

    ax[0].imshow(imgg, cmap="gray")
    ax[0].set_axis_off()
    ax[0].set_title("Original Image", fontsize=12)

    ax[1].imshow(cv[0], cmap="gray")
    ax[1].set_axis_off()
    title = "Chan-Vese segmentation - {} iterations".format(len(cv[2]))
    ax[1].set_title(title, fontsize=12)

    ax[2].imshow(cv[1], cmap="gray")
    ax[2].set_axis_off()
    ax[2].set_title("Final Level Set", fontsize=12)

    ax[3].plot(cv[2])
    ax[3].set_title("Evolution of energy over iterations", fontsize=12)

    fig.tight_layout()
    plt.show()


def getEstimation(center, center1, radius, radius1):
    errorCenter = abs(center1[0] - center[0]) + abs(center1[1] - center[1])
    errorR = abs(radius - radius1)
    if errorCenter < 2 and errorR < 3:
        estimation = Estimation.Super
    elif errorCenter + errorR < 10:
        estimation = Estimation.Satisfying
    else:
        estimation = Estimation.Bad
    return estimation


def cv_disk(image_size, center, radius):
    centerX, centerY = center
    """Generates a disk level set function.

    The disk covers the whole image along its smallest dimension.
    """
    res = np.ones(image_size)
    res[centerY, centerX] = 0.
    return (radius - distance(res)) / radius


def auto_canny(image, sigma=0.33):
        # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    np.set_printoptions(precision=2)
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    print('Missed sum:', cm[0, 1:].sum())
    print('Not recognized sum:', cm[1:, 0].sum())
    print('Recognized:', cm.diagonal()[1:].sum())
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot2Images(i1, i2):
    fig = plt.figure()
    plt.subplot(21)
    plt.imshow(i1)
    plt.subplot(22)
    plt.imshow(i2)
    plt.show()
    cv2.waitKey(0)


def plot4Images(i1, i2, i3, i4):
    fig = plt.figure()
    plt.subplot(221)
    plt.imshow(i1)
    plt.subplot(222)
    plt.imshow(i2)
    plt.subplot(223)
    plt.imshow(i3)
    plt.subplot(224)
    plt.imshow(i4)
    plt.show()
    cv2.waitKey(0)
