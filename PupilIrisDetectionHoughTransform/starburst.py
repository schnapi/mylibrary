import numpy as np
import cv2
import utilities as ut
import scipy
from skimage.measure import regionprops


def locate_corneal_reflection(I, cx, cy, window_width):

    # Input:
    # I = input image
    #[cx cy] = window center
    # window_width = width of window (must be odd)
    # Output:
    #[x,y] = corneal reflection coordinate
    # ar = approximate radius of the corneal reflection

    (height, width) = np.shape(I)
    i = 1
    score = [0]
    x = []
    y = []
    ar = []

    r = (window_width - 1) // 2
    sx = max(round(cx - r), 1)
    ex = min(round(cx + r), width)
    sy = max(round(cy - r), 1)
    ey = min(round(cy + r), height)
    Iw = I[sy:ey, sx:ex]
    (height, width) = np.shape(Iw)
    count = 0
    for threshold in np.arange(np.max(I), 0, -1):
        count += 1
        score.append(0)
        Iwt = np.zeros((height, width))
        Iwt[np.where(Iw >= threshold)] = 1
        s = [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]]
        labeled, numObjects = scipy.ndimage.measurements.label(Iwt, structure=s)
        if numObjects < 2:
            continue
        props = regionprops(labeled)
        max_area_index = max(range(len(props)), key=lambda i: props[i].area)
        sum_areas = sum(e.area for e in props)
        score[i] = props[max_area_index].area / (sum_areas - props[max_area_index].area)
        if score[i] - score[i - 1] < 0:
            y = props[max_area_index].centroid[0]
            x = props[max_area_index].centroid[1]
            ar = props[max_area_index].equivalent_diameter / 2
            break
        i = i + 1

    if x == []:
        return None

    x = x + sx - 1
    y = y + sy - 1
    return ut.roundToInt([x, y, ar])


def fit_circle_radius_to_corneal_reflection(I, crx, cry, crar, angle_delta):

    # This function uses a model-based minimization technique to find the radius at which the
    # corneal reflection intensity profile is approximately at 1 standard deviation. It
    # requires the center and approximate radius of the corneal reflection to run.

    # Input:
    # I = input image
    # [crx, cry] = location of the corneal reflection center
    # crar = approximate radius of the corneal reflection
    # angle_delta = angle step size
    #
    # Output:
    # r = optimized corneal reflection radius

    if crx == 0 or cry == 0 or crar == 0:
        return 0
    #r = scipy.optimize.minimize(circular_error, x0=crar, args=(I, angle_delta, crx, cry), method='Nelder-Mead')

    r = scipy.optimize.fmin(func=circular_error, args=(I, angle_delta, crx, cry), x0=crar)

    if len(r) <= 0:
        print('Error! the radius of corneal reflection is 0')
        return 0
    return r[0]


def circular_error(r, I, angle_delta, crx, cry):
    (height, width) = np.shape(I)
    r_delta = 1
    f = 0
    Isum = 0
    Isum2 = 0.000001
    m = np.arange(0, 2 * np.pi, angle_delta + 1)
    cos_m = np.cos(m)
    sin_m = np.sin(m)
    for i in np.arange(0, len(m)):
        x = crx + (r + r_delta) * cos_m[i]
        y = cry + (r + r_delta) * sin_m[i]
        x2 = crx + (r - r_delta) * cos_m[i]
        y2 = cry + (r - r_delta) * sin_m[i]
        if (x > 0 and y > 0 and x < width and y < height) and (x2 > 0 and y2 > 0 and x2 < width and y2 < height):
            Isum = Isum + I[ut.ceilToInt(y[0]), ut.ceilToInt(x[0])]
            Isum2 = Isum2 + I[ut.ceilToInt(y2[0]), ut.ceilToInt(x2[0])]
    f = (Isum / Isum2)
    return f


def remove_corneal_reflection(I, crx, cry, crr, angle_delta):
    # Input:
    # I = input image
    # angle_delta = angle step size
    # [crx cry] = corneal reflection center
    # crr = corneal reflection radius
    # Output:
    # I = output image with the corneal reflection removed

    if crx == 0 or cry == 0 or crr <= 0:
        return 0

    (height, width) = np.shape(I)

    if crx - crr < 1 or crx + crr > width or cry - crr < 1 or cry + crr > height:
        print('Error! Corneal reflection is too near the image border')
        return 0

    theta = np.arange(0, 2 * np.pi, np.pi / 360)
    tmat = np.matlib.repmat(theta, crr, 1).transpose()
    rmat = np.matlib.repmat(np.arange(1, crr + 1), len(theta), 1)
    (xmat, ymat) = ut.polar2cart(tmat, rmat)
    xv = np.reshape(xmat, (1, xmat.size))
    yv = np.reshape(ymat, (1, ymat.size))
    temp = ut.roundToInt(ymat[:, -1] + cry) + (ut.roundToInt(xmat[:, -1] + crx) - 1) * height
    Ir = I.ravel()[temp]
    avgmat = np.ones(rmat.shape) * np.mean(Ir)
    permat = np.matlib.repmat(Ir, crr, 1).transpose()
    wmat = np.matlib.repmat(np.arange(1, crr + 1) / crr, theta.size, 1)
    imat = avgmat * (1 - wmat) + permat * wmat
    imat = np.reshape(imat, (1, imat.size))
    I = I.ravel()
    I[ut.roundToInt(yv[0] + cry) + (ut.roundToInt(xv[0] + crx - 1) * height)] = imat[0]
    I = np.reshape(I, (height, width))
    cv2.imshow('edges', I)
    cv2.waitKey(0)


def detect_pupil_and_corneal_reflection(I, sx, sy, edge_thresh):
    #edge_thresh = 0
    # Input:
    # I = input image
    # [sx sy] = start point for starburst algorithm
    # edge_thresh = threshold for pupil edge detection
    #
    # Output:
    # pupil_ellipse = 5-vector of the ellipse parameters of pupil
    #   [a b cx cy theta]
    #   a - the ellipse axis of x direction
    #   b - the ellipse axis of y direction
    #   cx - the x coordinate of ellipse center
    #   cy - the y coordinate of ellipse center
    #   theta - the orientation of ellipse
    # cr_circle = 3-vector of the circle parameters of the corneal reflection
    #   [crx cry crr]
    #   crx - the x coordinate of circle center
    #   cry - the y coordinate of circle center
    #   crr - the radius of circle
    sigma = 2                      # Standard deviation of image smoothing
    angle_delta = 1 * np.pi / 180         # discretization step size (radians)
    cr_window_size = 101             # corneal reflection search window size (about [sx,sy] center)
    min_feature_candidates = 10      # minimum number of pupil feature candidates
    max_ransac_iterations = 10000    # maximum number of ransac iterations
    rays = 18

    I = cv2.GaussianBlur(I, (int(2.5 * sigma), int(2.5 * sigma)), sigma, sigma)

    [crx, cry, crar] = locate_corneal_reflection(I, sx, sy, cr_window_size)

    ut.drawCircle(I, (ut.roundToInt(crx), ut.roundToInt(cry)), ut.roundToInt(crar))
    crr = fit_circle_radius_to_corneal_reflection(I, crx, cry, crar, angle_delta)
    crr = ut.ceilToInt((crr * 2.5))

    I = remove_corneal_reflection(I, crx, cry, crr, angle_delta)
    # cr_circle = [crx cry crr];

    #[epx, epy] = starburst_pupil_contour_detection(I, sx, sy, edge_thresh, rays, min_feature_candidates);

    # if isempty(epx) || isempty(epy)
    #    pupil_ellipse = [0 0 0 0 0]';
    #    return;
    # end

    #[ellipse, inliers] = fit_ellipse_ransac(epx, epy, max_ransac_iterations);

    #[pupil_ellipse] = fit_ellipse_model(I, ellipse, angle_delta);

    # if ellipse(3) < 1 || ellipse(3) > size(I,2) || ellipse(4) < 1 || ellipse(4) > size(I,1):
    #    fprintf(1, 'Error! The ellipse center lies out of the image\n');
    #    pupil_ellipse = [0 0 0 0 0]';
    # return [pupil_ellipse, cr_circle]
