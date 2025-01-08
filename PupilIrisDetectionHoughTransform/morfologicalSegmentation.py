import cv2
import fill
import numpy as np
import utilities as ut
import scipy.ndimage as ndimage

# a) The average intensity of the sub image in the window must be less than a threshold T1.
# b) The pixels, which are placed in the corners and in the middle of the square sides, are critical pixels in the window, which must be resided in the iris, so the intensity of these eight pixels must be larger than a threshold T2.
# c) Minimum and maximum intensity of the pixels, which are placed in the margin of the window, is denoted by m and M, respectively. These values are calculated for each window and operation of finding the optimum window is stopped whenever the following inequalities are satisfied:,
# M . new old new old m m M > <(2)


def split_image(image, numofblocks=(4, 8)):
    threshold_subimage = 30
    threshold_cornes = 100
    # numofblocks tuple vertical, horizontal
    (H, W) = np.shape(image)
    (verticalH, horizontalW) = numofblocks
    win_vert = H // verticalH
    win_hori = W // horizontalW
    list_mean = []
    xlines = []
    ylines = []
    for k2 in range(0, verticalH):
        for k1 in range(0, horizontalW):
            win = image[k2 * win_vert:(k2 + 1) * win_vert, k1 * win_hori:(k1 + 1) * win_hori]
            list_mean += [np.mean(win)]
            xlines += [(k1 + 1) * win_hori - 1] * H
            ylines += range(0, H)
        xlines += range(0, W)
        ylines += [(k2 + 1) * win_vert - 1] * W
    ind = np.argmin(list_mean)
    # print(ind)
    (k2, k1) = np.unravel_index(ind, (verticalH, horizontalW))
    win = image[k2 * win_vert:(k2 + 1) * win_vert, k1 * win_hori:(k1 + 1) * win_hori]

    # minnew, minold
    # maxnew, maxold

    np.argmin(win)
    ut.DrawMarkImage(image, ylines, xlines)
    pass


def iterateDifferentWindowSize():
    pass


def detect_boundary(image, thresholdLow=60):
    threshold = 1.2 * np.mean(image[np.where(image < thresholdLow)])
    indices1 = np.where(image < threshold)
    binarized_image = np.zeros(image.shape, dtype=np.uint8)
    binarized_image[indices1] = 255

    edge_image = cv2.Canny(image, 80, 140, None, 3)
    img_dil = binarized_image.copy()
    fh = fill.slow_fill(edge_image, four_way=True).astype(bool)  # fill holes
    while(True):
        h = img_dil.copy()
        img_dil = ndimage.morphology.binary_dilation(img_dil)  # binarized image is not perfect and can contains coarse edges with reefs
        # ut.imshow(img_dil, 'dilated image')
        # fh is perfect circle but we need to remove noise around circle
        g = img_dil & fh  # logical and beetwen dilated image and image with circle and noise around circle
        # ut.imshow(fh, 'fill holes image')
        # ut.imshow(g, 'img dil: and :fill holes')
        if np.array_equal(g, h):
            break
        img_dil = g.copy()
    # g = ndimage.morphology.binary_erosion(g)
    g = g - ndimage.morphology.binary_erosion(g)
    ut.imshow(g, 'res')
    return g
