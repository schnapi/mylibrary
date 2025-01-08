import glob
from Enums import EyeDiseases, Databases
import cv2
import fileUtil
import houghTransform as ht
import kasaPupilDetection
import utilities as ut
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import fill
from scipy.stats import norm
import diseases_data as dd


REPLACE_PUPIL_WITH_IRIS_MEAN_TOLERANCE = 5
VERBOSE = -1  # -1 show just final segmentation, 1 shows some middle steps.
CANNY_FACTOR = 3  # canny low threshold is multiplied with CANNY_FACTOR, default is 3. for DISEASES data 2 is better
SELECTED_DATABASE = Databases.UTIRIS_database  # here we need to select data
PATH = "C:/Users/Sandi/Desktop/magistrska naloga/" + SELECTED_DATABASE.value  # path to database

if SELECTED_DATABASE == Databases.IITD_database:  # here we load database if exist, if not exist empty list are returned
    diseases_hand_checked, files = dd.IITD_database_checked("C:/Users/Sandi/Desktop/magistrska naloga/", SELECTED_DATABASE.name + '.csv')
    assert not EyeDiseases.exist('ArcusSenilis'), 'remove ArcusSenilis property (class EyeDiseases) in Enums.py for better results'
elif SELECTED_DATABASE == Databases.UTIRIS_database:
    diseases_hand_checked, files = dd.UTIRIS_database_checked("C:/Users/Sandi/Desktop/magistrska naloga/", SELECTED_DATABASE.name + '.csv')
    assert not EyeDiseases.exist('ArcusSenilis'), 'remove ArcusSenilis property (class EyeDiseases) in Enums.py for better results'
elif SELECTED_DATABASE == Databases.DISEASES:
    diseases_hand_checked, files = dd.diseases_checked(PATH + '/')
    assert EyeDiseases.exist('ArcusSenilis'), 'add ArcusSenilis property (class EyeDiseases) in Enums.py for better results'
    CANNY_FACTOR = 2

database_labeling = None  # if you want to label again set this to True or clean csv file
if files == []:  # if list is empty we read all images from database folder and start with labeling
    files = [file.replace('\\', '/') for file in glob.glob(PATH + '/[0-9]*/*.bmp', recursive=True)]  # find bmp and jpg
    files += [file.replace('\\', '/') for file in glob.glob(PATH + '/[0-9]*/*.jpg', recursive=True)]
    VERBOSE, database_labeling = 0, True
y_true, y_prediction = [ut.get_vector_of_diseases(sublist) for sublist in diseases_hand_checked], []  # check function description
diseasesTP, diseasesFP, diseasesFN = [], [], []
flag, index = True, 0
for file in files:
    # if not file.endswith('010/04_L.bmp') and flag:
    #     index += 1
    #     continue
    # flag = False
    print(file)
    img_color = ut.resize_image(cv2.imread(file, 1), 320, "width")  # resize image on fix dimension
    img_hsv, img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV), cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    cv2.normalize(img_gray, img_gray, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)  # image normalization
    img_equalized = cv2.equalizeHist(img_gray)
    img_no_reflection = cv2.bitwise_not(fill.slow_fill(cv2.bitwise_not(img_equalized), four_way=True))  # fill holes, reflection becomes more dark
    if img_no_reflection.max() == 1:  # this is slow_fill bug
        img_no_reflection = img_equalized
    # Gaussian blurring is a linear operation. However, it does not preserve edges in the input image - the value of sigma governs the degree of smoothing, and eventually how the edges are preserved.
    # The Median filter is a non-linear filter. Unlike linear filters, median filters replace the pixel values with the median value available in the local neighborhood (say, 5x5 or 3x3 pixels around the central pixel value). Also, median filter is edge preserving (the median value must actually be the value of one of the pixels in the neighborhood). This is probably a good read: http://arxiv.org/pdf/math/061242...
    # Bilateral filter is a non-linear filter. It prevents averaging across image edges, while averaging within smooth regions of the image  -> hence edge-preserving. Also, Bilateral filters are non-iterative.
    # Apply a Gaussian blur to reduce noise and avoid false circle detection: GaussianBlur(img,kernelSize,Standard deviation in x direction, in y)
    # blurred = cv2.GaussianBlur(img, (9,9), 2,2)
    img_medianblured = cv2.medianBlur(img_equalized, 9)  # 9-window size, best filter for proposed image, reduce noise
    # canny (img,Hysteresis,H,size of sobel kernel); Hysteresis=>http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html
    # othersSegmentationMethods.try_all_threshold_test(file)

    # 8
    center_pupil_kasa, radius_pupil_kasa, compactness_kasa = kasaPupilDetection.kasa(img_no_reflection)  # kasa method is applied
    # 9
    eucl_dist, step = 1000, 1.1  # iris detection
    while step < 1.3:  # step is multiplied with radius_pupil_kasa, this is minimum radius of iris circle
        radius_iris_in, center_iris_in, radius_iris_out, center_iris_out, canny_thresh = ht.detect_circle_iris(img_medianblured, center_pupil_kasa,
                                                                                                               radius_pupil_kasa * step, CANNY_FACTOR,
                                                                                                               canny_low_thresh=80,
                                                                                                               euclidean_limit_dist=eucl_dist)
        if compactness_kasa > 0.8 and radius_pupil_kasa > radius_iris_out / 2 and \
                ut.is_circle_outside(center_pupil_kasa, radius_pupil_kasa, center_iris_out, radius_iris_out):
            eucl_dist = 11
            step += .1
        else:
            break
    # ut.draw3Circle(img_gray, center_iris_out, radius_iris_out, center_iris_in, radius_iris_in, center_pupil_kasa, radius_pupil_kasa, verbose=verbose)

    # what if iris is pupil (problem with strong edges around pupil)
    # 10
    if ut.is_iris_pupil(img_color, img_hsv, center_iris_out, radius_iris_out):
        _, _, radius_iris_out, center_iris_out, canny_thresh = ht.detect_circle_iris(
            img_medianblured, center_iris_out, radius_iris_out * 1.1, CANNY_FACTOR, canny_low_thresh=canny_thresh)  # we need new bigger iris
        if not ut.is_mydriasis_miosis(radius_iris_in, radius_iris_out):  # if radius_iris_in is normal then we take radius_iris_in
            center_pupil, radius_pupil = center_iris_in, radius_iris_in
        else:  # we choose radius_iris_out
            center_pupil, radius_pupil = center_iris_out, radius_iris_out
        ut.draw3Circle(img_gray, center_iris_out, radius_iris_out, center_iris_in,
                       radius_iris_in, center_pupil, radius_pupil, 'test2', verbose=VERBOSE)
    # 11
    center_pupil, radius_pupil, _, pupil_convexhull, compactness = ut.gaussian_mixture_model_segmentation(
        img_hsv, img_gray, center_iris_out, radius_iris_out, verbose=VERBOSE)  # backup if kasa method fail
    # what is better, result from kasa or gaussian mixture model
    # 12
    if center_pupil_kasa and ut.is_circle_inside(center_pupil_kasa, radius_pupil_kasa, center_iris_out, radius_iris_out, safe_factor=1):
        # here we calculate probabilities for replacing new pupil with old. compactness? now is linear from 1(circle), 0.75(square), ...
        # maybe we can use triangular distribution instead normal (then we are more sensitive to small changes in peaks)
        std_dist, std_radius = radius_iris_out / 8, radius_iris_out / 2  # in that case distance is more important (std is lower)
        p1_dist = norm.pdf(ut.euclidean_distance(center_iris_out, center_pupil), 0, std_dist)  # p(how close we are to iris center)
        p2_dist = norm.pdf(ut.euclidean_distance(center_iris_out, center_pupil_kasa), 0, std_dist)
        p1_dist /= norm.pdf(0, 0, std_dist)  # normalization
        p2_dist /= norm.pdf(0, 0, std_dist)
        p1_radius = norm.pdf(radius_pupil, radius_iris_out / 2, std_radius)  # p(how far away we are from normal size of the pupil)
        p2_radius = norm.pdf(radius_pupil_kasa, radius_iris_out / 2, std_radius)  # normal size of pupil is radius_iris_out / 2???
        p1_radius /= norm.pdf(0, 0, std_radius)
        p2_radius /= norm.pdf(0, 0, std_radius)
        p1, p2 = p1_radius * compactness * p1_dist, p2_radius * compactness_kasa * p2_dist
        if VERBOSE > 0:
            print('-' * 50, 'p(gaussian mm):', p1 / max(p1, p2) * 100, 'p(kasa):', p2 / max(p1, p2) * 100)
        if p1 < p2:
            radius_pupil, center_pupil, pupil_convexhull = radius_pupil_kasa, center_pupil_kasa, None  # we replace new found pupil with old one
    # with radius_iris_in > radius_pupil we prevent degradation of pupil
    # e.g. if iris_in is smaller than pupil -> in next step we can't replace pupil with iris)
    if radius_iris_in < radius_pupil:
        center_iris_in, radius_iris_in = None, 0
    # 13
    if center_iris_in:  # maybe iris_in is pupil
        # first we check if pupil is outside iris, something is wrong with pupil or iris
        # we also compare means of area and if are very similiar then pupil is iris
        if ut.is_circle_outside(center_pupil, radius_pupil, center_iris_in, radius_iris_in, safe_factor=0.90) \
                or ut.get_circle_pixels_values(img_no_reflection, center_pupil, radius_pupil).mean() + REPLACE_PUPIL_WITH_IRIS_MEAN_TOLERANCE > \
                ut.get_circle_pixels_values(img_no_reflection, center_iris_in, radius_iris_in).mean() and \
                ut.euclidean_distance(center_pupil, center_iris_in) > REPLACE_PUPIL_WITH_IRIS_MEAN_TOLERANCE * 2:  # if no big difference in color
            radius_pupil, center_pupil = radius_iris_in, center_iris_in  # we replace pupil with iris_in circle
            center_iris_in, radius_iris_in = None, 0
            pupil_convexhull = None
    # radius_iris_out < radius_pupil -> we dont want to replace smaller pupil with bigger noisy iris. we do that if mean of area diff is small enough
    # 14
    if ut.is_circle_outside(center_pupil, radius_pupil, center_iris_out, radius_iris_out, safe_factor=1.1) and radius_iris_out < radius_pupil \
            or ut.get_circle_pixels_values(img_no_reflection, center_pupil, radius_pupil).mean() + REPLACE_PUPIL_WITH_IRIS_MEAN_TOLERANCE > \
            ut.get_circle_pixels_values(img_no_reflection, center_iris_out, radius_iris_out).mean():
        radius_pupil, center_pupil = radius_iris_out, center_iris_out  # new iris is find by find new iris function
        pupil_convexhull = None
    # 15
    radius_iris_in, center_iris_in, radius_iris_out, center_iris_out = ut.find_new_iris(img_equalized, img_medianblured,
                                                                                        radius_iris_out, center_iris_out,
                                                                                        center_pupil, radius_pupil, center_iris_in,
                                                                                        radius_iris_in, canny_thresh, CANNY_FACTOR)
    # 16
    diseases = ut.detect_diseases_in_iris(img_gray, img_medianblured, center_pupil, radius_pupil, center_iris_out, radius_iris_out,
                                          center_iris_in, radius_iris_in, pupil_convexhull, VERBOSE)
    #  -------------------------------------------------------------------------------------------------------------
    # if EyeDiseases.ArcusSenilis not in diseases:  # clean
    #     center_iris_in, radius_iris_in = None, 0
    diseases_vector = set(diseases_hand_checked[index]) if len(diseases_hand_checked) > index else set()
    y_prediction.append(ut.get_vector_of_diseases(diseases))  # y_true[index]
    intersection = list(diseases & diseases_vector)
    if intersection != []:
        diseasesTP.append((file, ' Bolezni, ki so pravilno zaznane: ', intersection))
        if VERBOSE > 0:
            print('Bolezni, ki so pravilno zaznane: ' + str(intersection))
    diff = list(diseases - diseases_vector)
    if diff != []:
        diseasesFP.append((file, ' Bolezni, ki so napacno zaznane: ', diff))
        if VERBOSE > 0:
            print('Bolezni, ki so napacno zaznane: ' + str(diff), '-' * 40)
    diff = list(diseases_vector - diseases)
    if diff != []:
        diseasesFN.append((file, ' Bolezni, ki niso zaznane: ', diff))
        if VERBOSE > 0:
            print('Bolezni, ki niso zaznane: ' + str(diff), '-' * 40)
    # print(radius_pupil, center_pupil, radius_iris_out, center_iris_out)
    ut.draw3Circle(img_color, center_iris_out, radius_iris_out, center_iris_in, radius_iris_in,
                   center_pupil, radius_pupil, '/'.join(fileUtil.get_file_name(file)), coords=pupil_convexhull,
                   label=diseases, verbose=1 if VERBOSE in [-1, 1] else 0)
    if database_labeling is not None:
        dd.labeling_database(file, img_gray, center_iris_out, radius_iris_out, center_iris_in, radius_iris_in,
                             center_pupil, radius_pupil, diseases, SELECTED_DATABASE)
    cv2.destroyAllWindows()
    # (xP, yP, (a, h, k)) = ht.getParabola(edges, center_iris_out, radius_iris_out, -1)
    # yC, xC = ut.getPointsInCircle(center_iris_out, radius_iris_out, centerPupil, radiusPupil, (H, W))
    # # ut.DrawMarkImage(imgGray,yC,xC)

    # (y1, x1), _ = ut.getPointsBelowAboveParabola(yC, xC, a, h, k + 0.4 * radius_iris_out)

    # # (yL,xL) = ut.getLowerHalfOfCircle(y1,x1,centerIris)
    # y1, x1 = ut.removePixelsBasedOnColorValue(img, y1, x1, DarkestPixelValue=20)

    # # hist, bin_edges = np.histogram(imgGray[y1,x1],255)
    # # dad = np.argmax(hist)
    # # assd = np.max(hist)
    # imgColor = ut.MarkImage(img, yP, xP)
    # ut.DrawMarkImage(imgColor, y1, x1)

    # ut.DrawMarkImage(imgGray,y,x)
    index += 1

print("\n-------------------------------------------------")
[print(file_ + reason + str(disease)) for file_, reason, disease in diseasesFP]
print("\n-------------------------------------------------")
[print(file_ + reason + str(disease)) for file_, reason, disease in diseasesFN]

y_true = [i for g in y_true for i in g]
y_prediction = [i for g in y_prediction for i in g]
cnf_matrix = confusion_matrix(y_true, y_prediction, labels=EyeDiseases.diseases_table())
plt.figure()
ut.plot_confusion_matrix(cnf_matrix, classes=EyeDiseases.diseases_table(),
                         title='Matrika zmot ({:.0f} primerkov)'.format(cnf_matrix.sum() / len(EyeDiseases)))
plt.show()
pass
