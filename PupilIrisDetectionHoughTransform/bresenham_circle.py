def circle(radius):
    "Bresenham complete circle algorithm in Python"
    # init vars
    switch = 3 - (2 * radius)
    points = set()
    x = 0
    y = radius
    # first quarter/octant starts clockwise at 12 o'clock
    while x <= y:
        # first quarter first octant
        points.add((x, -y))
        # first quarter 2nd octant
        points.add((y, -x))
        # second quarter 3rd octant
        points.add((y, x))
        # second quarter 4.octant
        points.add((x, y))
        # third quarter 5.octant
        points.add((-x, y))
        # third quarter 6.octant
        points.add((-y, x))
        # fourth quarter 7.octant
        points.add((-y, -x))
        # fourth quarter 8.octant
        points.add((-x, -y))
        if switch < 0:
            switch = switch + (4 * x) + 6
        else:
            switch = switch + (4 * (x - y)) + 10
            y = y - 1
        x = x + 1
    return list(points)


import cv2
import numpy as np
import utilities as ut
# img = np.array([[0, 3, 0], [1, 1, 1], [0, 2, 0]])


def test(img):
    b = img.copy()
    h, w = b.shape
    limit = 255
    center = b.shape[0] // 2
    center2 = np.array([center, center])
    r = 10
    c1 = (0, 0)
    while r < center:
        b = img.copy()
        a = np.array(circle(r)) + center2
        y, x = a[:, 1], a[:, 0]
        temp = max(np.min(b[y, x]), 10)
        temp = min(limit, temp)
        b[np.where(b > limit)] = 255
        limit = temp
        print(ut.crop_image(b, center2, r))
        r += 1
    print(limit)
    ut.drawCircle(b, center2, r)
