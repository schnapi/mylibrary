import cv2
import numpy as np
import utilities as ut


def detect_circle_iris(image, iris_center, min_radius, canny_factor, canny_low_thresh=80, euclidean_limit_dist=11, max_radius=None):
    """Returns two circles. Idea is one for iris and another for pupil or circle inside iris which is an indicator of disease.

    Params:
    ------
    canny_low_thresh: canny low threshold for HoughCircles method.
    euclidean_limit_dist: set this to 1000 if center is not known, in that case euclidean_distance returns 1000 or less (we skip continue)
    """
    mis_dist_between_circles, accumulator_thresh, canny_low_limit = 2, 40, 30  # parameters for houghcircles
    arcus_senilis_range = 0.10  # 15% away from found radius
    if canny_low_thresh <= canny_low_limit:
        canny_low_thresh = 40
    canny_low_thresh_best = canny_low_thresh
    accumulator_thresh_flag = True  # when we find iris we stop reducing accumulator_thresh -> we dont want to find fake circles
    if max_radius is None:
        max_radius = min(image.shape) // 2
    min_radius, max_radius = ut.roundToInt([min_radius, max_radius])
    center, radius = None, 0
    while canny_low_thresh > canny_low_limit:
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, mis_dist_between_circles, param1=canny_low_thresh * canny_factor,
                                   param2=accumulator_thresh, minRadius=min_radius, maxRadius=max_radius)
        if circles is not None:
            # edges = cv2.Canny(image, canny_low_thresh, canny_low_thresh * 2, None, 3)  # note canny is implemented in hough transform
            # ut.imshow3(edges, 'k', 1)
            if circles.size > 1000 and center is not None:  # too many circles mean too much noise and wrong circles
                return 0, None, radius, center, canny_low_thresh_best
            for x, y, r in np.around(circles[0]).astype(int):
                if not ut.inimage((x, y), r, image) or ut.euclidean_distance(iris_center, [x, y]) > euclidean_limit_dist:
                    continue  # if circle is outside image or euclidean_dist(iris_center, (x,y)) > our limit
                if center is None:  # if circle has not been found yet
                    center, radius = tuple(ut.roundToInt((x, y))), ut.roundToInt(r)  # circle found
                    # ut.drawCircle(image, center, radius)
                    iris_center = center  # new iris center
                    euclidean_limit_dist = r / 10  # 61/10 = 6.1 -> in this range we search next circle center
                    canny_low_thresh_best = canny_low_thresh  # best thresh to return for next iterations
                    canny_low_limit = ut.roundToInt(canny_low_thresh * .66)  # reduce for 2/3
                    accumulator_thresh_flag = False  # enough edges
                    accumulator_thresh = 40  # reset because of new iris search
                    continue
                diff = abs(radius - r)
                # if diff is < radius * arcus_senilis_range -> 61 * .15 = 9.15
                # we don't want find to circles too close to each other
                if diff < radius * arcus_senilis_range:  # or diff > radius * 2.40: # or
                    continue
                if r < radius:  # iris out is bigger so replace
                    return ut.roundToInt(r), tuple(ut.roundToInt((x, y))), radius, center, canny_low_thresh_best
                return radius, center, ut.roundToInt(r), tuple(ut.roundToInt((x, y))), canny_low_thresh_best
        canny_low_thresh -= 5  # more edges
        if accumulator_thresh_flag:
            accumulator_thresh -= 5
            if accumulator_thresh <= 0:  # this is in case if we couldn't find edges for some canny low threshold. We just reset accu.
                accumulator_thresh = 40  # reset sensitivity
    return 0, None, radius, center, canny_low_thresh_best


def getParabola(edges, center, radius, sign=-1):
    stepA = 0.001  # za koliko ukrivljamo parabolo v posameznem koraku
    coffToIndex = 1 / stepA  # pretvorba v index
    maxCurvature = 0.004  # max ukrivljenost
    # h - x axis parabola, k - y axis parabola } vertex - goriščna točka
    aRange = np.arange(stepA, maxCurvature, stepA)
    hMin = center[0] - radius
    if(hMin < 0):
        hMin = 0
    hRange = np.arange(hMin, center[0] + radius)  # območje goriščne točke parabole je omejeno od centra šarenice po x osi za radij v levo in desno stran
    # območje goriščne točke parabole je omejeno od centra šarenice po y osi za centerY-1.5*radij do centerY-0.3*radij
    kMax = int(round(center[1] - 0.3 * radius))
    kMin = int(round(center[1] - 1.5 * radius))
    # Accumulator size: a,h,k
    Accumulator = np.zeros((len(aRange), len(hRange), kMax - kMin), dtype=np.int)
    (yEdges, xEdges) = np.where(edges)
    # enačba parabole: y = a(x-h)^2+k
    for a in aRange:
        for h in hRange:
            # y(i) change sign due to y axis is reversed
            k = -a * (xEdges - h)**2 + (yEdges)
            k = np.round(k).astype(int)
            k = k[np.where(k < kMax)]
            k = k[np.where(k > kMin)]
            # a-1 we start with a*coffToIndex== 1
            np.add.at(Accumulator[int(a * coffToIndex) - 1, h - hMin], k - kMin, 1)
    sorted = np.argsort(Accumulator.ravel())
    [H, W] = np.shape(edges)
    xParabola = np.arange(0, W)
    # spremenljivke za izbrano parabolo
    MinNumOfPixels = ut.getMax(int)
    selected = None
    a1 = 0
    h1 = 0
    k1 = 0
    # get best 34 and from them get the one with lowest number of pixEdgesls below parabola
    for i in np.arange(-1, -35, -1):
        # dobimo parametre
        (a, h, k) = np.unravel_index(sorted[i], Accumulator.shape, 'C')
        # jih vstavimo, da dobimo y
        yParabola = np.round((a + 1) / coffToIndex * (xParabola - (h + hMin))**2 + k + kMin).astype(int)
        # štetje pikslov na levi in desni strani, da dobimo najnižjo parabolo
        # vzamemo točke na sliki, nižje od parabole in višje od centra
        indices = ut.getIndicesBelowParabolaAndAboveLine(yEdges, xEdges, a + 1, h + hMin, k + kMin, center[1])
        leftInd = np.where(xEdges[indices] <= W / 2)  # točke na levi strani
        sumResLeft = edges[yEdges[leftInd], xEdges[leftInd]].sum()  # vsota točk na levi strani
        rightInd = np.where(xEdges[indices] > W / 2)  # točke na desni strani
        sumResRight = edges[yEdges[rightInd], xEdges[rightInd]].sum()  # vsota točk na desni strani
        # im=ut.MarkImage(edges,yEdges[indices],xEdges[indices],str(sumRes))
        # število točk na levi in desni strani mora biti vsaj 40, in vsota levih in desnih mora biti manjša od prejšnje vsote
        if sumResLeft / 255 > 40 and sumResRight / 255 > 40 and sumResLeft + sumResRight < MinNumOfPixels:
            MinNumOfPixels = sumResLeft + sumResRight
            selected = (yParabola, xParabola)
            a1 = a
            h1 = h
            k1 = k
        # ut.DrawMarkImage(im,yParabola,x)
    if selected is None:  # če ni izbrane parabola vzamemo prvo
        selected = (yParabola, xParabola)
        a1 = a
        h1 = h
        k1 = k
    else:
        (yParabola, xParabola) = selected
    # koeficente normaliziramo (a1+1,...)
    indices = np.where(yParabola < H)
    return (xParabola[indices], yParabola[indices], (a1 + 1, h1 + hMin, k1 + kMin))
