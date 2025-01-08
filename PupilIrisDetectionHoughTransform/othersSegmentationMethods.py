import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, segmentation, color
from skimage.future import graph
from skimage.color import rgb2gray, rgb2hsv
from skimage.segmentation import felzenszwalb, mark_boundaries, slic, random_walker
from skimage.transform import resize
from scipy import ndimage
from skimage.morphology import watershed, disk
from skimage.filters import rank
import houghTransform as ht
import utilities as ut
from sklearn.cluster import spectral_clustering


def kmeans(img, K=3):
    z = img.ravel()
    z = np.float32(z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1.0)
    ret, label, center = cv2.kmeans(z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    # cv2.imshow('res-' + str(K), res2)
    # cv2.waitKey(0)
    return res2, sorted([c[0] for c in center])


def kmeans_test(img):
    z = img.ravel()
    if img.ndim == 3:
        z = img.reshape((-1, 3))

    # convert to np.float32
    z = np.float32(z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1.0)
    for K in (2, 3, 4, 5, 6, 7, 8):

        ret, label, center = cv2.kmeans(z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))

        cv2.imshow('res-' + str(K), res2)
    cv2.waitKey(0)


def slic_test(image):
    # loop over the number of segments
    for numSegments in (2, 3, 4, 8, 44, 66, 100, 150, 200, 250):
        # apply SLIC and extract (approximately) the supplied number
        # of segments
        segments = slic(image, n_segments=numSegments, compactness=30.)

        # show the output of SLIC
        fig = plt.figure("Superpixels -- %d segments" % (numSegments))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mark_boundaries(image, segments))
        plt.axis("off")

    # show the plots
    plt.show()


def AgglomerativeClustering_test(img, n_clusters):

    from sklearn.feature_extraction.image import grid_to_graph
    from sklearn.cluster import AgglomerativeClustering
    X = np.reshape(img, (-1, 1))
    connectivity = grid_to_graph(*img.shape)

    # #############################################################################
    # Compute clustering
    print("Compute structured hierarchical clustering...")
    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward',
                                   connectivity=connectivity)
    ward.fit(X)
    label = np.reshape(ward.labels_, img.shape)
    print("Number of pixels: ", label.size)
    print("Number of clusters: ", np.unique(label).size)

    # #############################################################################
    # Plot the results on an image
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap=plt.cm.gray)
    for l in range(n_clusters):
        plt.contour(label == l, contours=1,
                    colors=[plt.cm.spectral(l / float(n_clusters)), ])
    plt.xticks(())
    plt.yticks(())
    plt.title(str(n_clusters))
    plt.show()


def spectral_clustering_test(img):
    img = ut.resize_image(img, 50, "width")
    from sklearn.feature_extraction import image

    # Convert the image into a graph with the value of the gradient on the
    # edges.
    graph = image.img_to_graph(img)

    # Take a decreasing function of the gradient: an exponential
    # The smaller beta is, the more independent the segmentation is of the
    # actual image. For beta=1, the segmentation is close to a voronoi
    beta = 5
    eps = 1e-6
    graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps

    # Apply spectral clustering (this step goes much faster if you have pyamg
    # installed)
    N_REGIONS = 22
    labels = spectral_clustering(graph, n_clusters=N_REGIONS, assign_labels='discretize', random_state=1)
    labels = labels.reshape(img.shape)

    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap=plt.cm.gray)
    for l in range(N_REGIONS):
        plt.contour(labels == l, contours=1,
                    colors=[plt.cm.spectral(l / float(N_REGIONS))])
    plt.xticks(())
    plt.yticks(())
    plt.show()


def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.node[dst]['total color'] += graph.node[src]['total color']
    graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
    graph.node[dst]['mean color'] = (graph.node[dst]['total color'] /
                                     graph.node[dst]['pixel count'])


def slic_and_merge(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    labels = segmentation.slic(img, compactness=30, n_segments=20)
    g = graph.rag_mean_color(img, labels)

    labels2 = graph.merge_hierarchical(labels, g, thresh=35, rag_copy=False,
                                       in_place_merge=True,
                                       merge_func=merge_mean_color,
                                       weight_func=_weight_mean_color)
    temp = np.max(labels2)
    # print(temp)
    out = color.label2rgb(labels2, img, kind='avg')
    out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))
    io.imshow(out)
    io.show()


def weight_boundary(graph, src, dst, n):
    """
    Handle merging of nodes of a region boundary region adjacency graph.

    This function computes the `"weight"` and the count `"count"`
    attributes of the edge between `n` and the node formed after
    merging `src` and `dst`.


    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the "weight" and "count" attributes to be
        assigned for the merged node.

    """
    default = {'weight': 0.0, 'count': 0}

    count_src = graph[src].get(n, default)['count']
    count_dst = graph[dst].get(n, default)['count']

    weight_src = graph[src].get(n, default)['weight']
    weight_dst = graph[dst].get(n, default)['weight']

    count = count_src + count_dst
    return {
        'count': count,
        'weight': (count_src * weight_src + count_dst * weight_dst) / count
    }


def merge_boundary(graph, src, dst):
    """Call back called before merging 2 nodes.

    In this case we don't need to do any computation here.
    """
    pass


from skimage.filters import sobel


def slic_and_merge1(img):
    img = img / 255.
    edges = sobel(img)
    labels = segmentation.slic(img, compactness=30, n_segments=400)
    g = graph.rag_boundary(labels, edges)

    graph.show_rag(labels, g, img)
    plt.title('Initial RAG')

    labels2 = graph.merge_hierarchical(labels, g, thresh=0.01, rag_copy=False,
                                       in_place_merge=True,
                                       merge_func=merge_boundary,
                                       weight_func=weight_boundary)

    graph.show_rag(labels, g, img)
    plt.title('RAG after hierarchical merging')

    plt.figure()
    out = color.label2rgb(labels2, img, kind='avg')
    plt.imshow(out)
    plt.title('Final segmentation')

    plt.show()


def read_image(path, shape=None, resize_width=320):
    image = io.imread(path)
    H, W, __ = image.shape
    if shape is not None:
        return resize(image, shape, mode='reflect')
    if W > resize_width:
        k = resize_width / W
        return resize(image, (ut.roundToInt(H * k), ut.roundToInt(W * k)), mode='reflect')
    return image


from skimage import exposure
import skimage


def watershed_pupil(path, imgGray):
    image1 = read_image(path, shape=imgGray.shape)
    gray = rgb2gray(image1)
    image = rgb2hsv(image1)
    image = image[:, :, 1]

    image = exposure.equalize_hist(image)
    # denoise image
    denoised = rank.median(image, disk(2))
    # find continuous region (low gradient) --> markers
    markers = rank.gradient(denoised, disk(5)) < 70
    markers = ndimage.label(markers)[0]
    # ut.imshow((markers).astype(np.uint8), 'markers')
    bw = np.zeros(imgGray.shape).astype(np.uint8)
    bw[np.where(imgGray < imgGray.min() + 10)] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)  # morfological opening! remove small regions

    a, contours, hier = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Initialize empty list
    ind = 255
    for i in range(len(contours)):
        if len(contours[i]) < 20:
            continue
        cimg = np.zeros_like(gray)
        cv2.drawContours(cimg, contours, i, color=255, thickness=-1)
        if ind in markers:
            markers[np.where(markers == ind)] = 0
        pts = np.where(cimg == 255)
        markers[pts[0], pts[1]] = ind
        ind -= 1
    # local gradient
    gradient = rank.gradient(imgGray / 255, disk(2))
    gradient = cv2.Canny(imgGray, 0, 2, None, 3)
    # process the watershed
    labels = watershed(gradient, markers)
    res = skimage.measure.regionprops(labels)
    showLabelsWatershed(image1, gradient, markers, gray, labels)

    res = [reg for reg in res if reg.area > 300 and reg.eccentricity < 0.9 and reg.area < np.prod(gray.shape) * 0.1]
    try:
        circle = min(res, key=lambda a: np.mean(gray[np.where(labels == a.label)]))
    except ValueError:
        return 0, None

    # (rmse, surface, center, radius) = ut.getContourBoundaryInfo(res.coords)
    # gray = showLabels(gray, res, labels)
    return (ut.center_bbox(circle.bbox), (circle.centroid[1], circle.centroid[0]))


def watershed_test(path):
    image1 = read_image(path)
    gray = rgb2gray(image1)
    image = rgb2hsv(image1)
    image = image[:, :, 1]

    image = exposure.equalize_hist(image)
    # denoise image
    denoised = rank.median(image, disk(2))
    # find continuous region (low gradient) --> markers
    markers = rank.gradient(denoised, disk(5)) < 30
    markers = ndimage.label(markers)[0]

    # local gradient
    gradient = rank.gradient(denoised, disk(2))

    # process the watershed
    labels = watershed(gradient, markers)
    res = skimage.measure.regionprops(labels)

    max_index, max_value = max(enumerate(res), key=lambda a: a[1].area)
    #  max(node.area for node in res)
    gray[np.where(labels == max_value.label)] = 1
    # display results
    showLabelsWatershed(image1, gradient, markers, gray, labels)
    (radius, center) = ht.detect_circle_pupil(labels.astype(np.uint8))
    gray = (gray * 255).astype(np.uint8)
    ut.drawCircle(gray, center, radius, "Pupil detection: ")


def showLabelsWatershed(image1, gradient, markers, gray, labels):
    fig, axes = plt.subplots(ncols=4, figsize=(8, 2.7))
    ax0, ax1, ax2, ax3 = axes

    ax0.imshow(image1, cmap=plt.cm.gray, interpolation='nearest')
    ax1.imshow(gradient, cmap=plt.cm.spectral, interpolation='nearest')
    ax2.imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')
    ax3.imshow(gray, cmap=plt.cm.gray, interpolation='nearest')
    ax3.imshow(labels, cmap=plt.cm.spectral, interpolation='nearest', alpha=.7)
    np.min(labels)
    for ax in axes:
        ax.axis('off')

    fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
    plt.show()


def showLabels(gray, res, labels):
    gray = (gray * 255).astype(np.uint8)
    for value in res:
        indices = np.where(labels == value.label)
        temp = gray.copy()
        print(value.bbox)
        print(value.centroid)
        print(np.mean(temp[indices]))
        print(value.eccentricity)
        print(value.area)
        temp[indices] = 255
        ut.imshow(temp)
        # ut.drawImageMouseEvent(temp)
    return gray


def min_threshold_test(path):
    from skimage.filters import threshold_minimum

    image = rgb2gray(read_image(path))

    thresh_min = threshold_minimum(image)
    binary_min = image > thresh_min

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    ax[0, 0].imshow(image, cmap=plt.cm.gray)
    ax[0, 0].set_title('Original')

    ax[0, 1].hist(image.ravel(), bins=256)
    ax[0, 1].set_title('Histogram')

    ax[1, 0].imshow(binary_min, cmap=plt.cm.gray)
    ax[1, 0].set_title('Thresholded (min)')

    ax[1, 1].hist(image.ravel(), bins=256)
    ax[1, 1].axvline(thresh_min, color='r')

    for a in ax[:, 0]:
        a.axis('off')
    plt.show()


def try_all_threshold_test(path):
    from skimage.filters import try_all_threshold
    img = rgb2gray(read_image(path))
    fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
    plt.show()


def otsu_extended(image):
    data = image.ravel()
    data = data
    hist_val, hist_ind = np.histogram(data, 100)
    hist_width = hist_ind[1] - hist_ind[0]
    hist_ind = hist_ind[:-1]

    seg_var_total = np.zeros(hist_ind.shape)

    values = hist_ind + hist_width / 2.
    counts = hist_val

    seg_var_total[0] = (values**2 * counts).sum() / counts.sum()
    seg_var_total[0] -= ((values * counts).sum() / counts.sum())**2
    for i in range(1, values.shape[0]):
        seg1_mean = (values[:i] * counts[:i]).sum() / counts[:i].sum()
        seg1_sq_mean = (values[:i]**2 * counts[:i]).sum() / counts[:i].sum()

        seg2_mean = (values[i:] * counts[i:]).sum() / counts[i:].sum()
        seg2_sq_mean = (values[i:]**2 * counts[i:]).sum() / counts[i:].sum()

        seg1_var = seg1_sq_mean - seg1_mean**2
        seg2_var = seg2_sq_mean - seg2_mean**2
        seg_var_total[i] = seg1_var + seg2_var

    otsu_prag = values[np.argmin(seg_var_total)]
    return otsu_prag


def otsu_test(path):
    from skimage.filters import threshold_otsu

    image = rgb2gray(read_image(path))
    thresh = threshold_otsu(image)
    binary = image > thresh

    fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
    ax = axes.ravel()
    ax[0] = plt.subplot(1, 3, 1, adjustable='box-forced')
    ax[1] = plt.subplot(1, 3, 2)
    ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0], adjustable='box-forced')

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Original')
    ax[0].axis('off')

    ax[1].hist(image.ravel(), bins=256)
    ax[1].set_title('Histogram')
    ax[1].axvline(thresh, color='r')

    ax[2].imshow(binary, cmap=plt.cm.gray)
    ax[2].set_title('Thresholded')
    ax[2].axis('off')

    plt.show()


def random_walker_test(path):
    data = rgb2gray(read_image(path))

    # convert to np.float32
    data = np.float32(data)
    markers = np.zeros(data.shape, dtype=np.uint)
    markers[data < 0.15] = 1
    markers[data > 0.95] = 2
    import utilities as ut
    labels = random_walker(data, markers, beta=10, mode='bf')
    labels[np.where(labels > 1)] = 255
    labels = np.uint8(labels)
    ut.imshow(labels)
    fig = plt.figure("random_walker")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(labels, cmap='gray', interpolation='nearest')
    plt.axis("off")
    plt.show()


def graph_based_segmentation_test(img):
    # loop over the number of segments
    for numSegments in (30,):
        # apply SLIC and extract (approximately) the supplied number
        # of segments
        labels = felzenszwalb(img, scale=numSegments, min_size=20)

        # show the output of SLIC
        fig = plt.figure("felzenszwalb -- %d segments" % (numSegments))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mark_boundaries(img, labels))
        plt.axis("off")
        g = graph.rag_mean_color(img, labels)

        labels2 = graph.merge_hierarchical(labels, g, thresh=35, rag_copy=False,
                                           in_place_merge=True,
                                           merge_func=merge_mean_color,
                                           weight_func=_weight_mean_color)

        out = color.label2rgb(labels2, img, kind='avg')
        out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))
        io.imshow(out)
        io.show()
    # show the plots
    plt.show()


from sklearn.mixture import GaussianMixture, BayesianGaussianMixture


def get_n_components(data, verbose=0):
    n_components = np.arange(1, 6)
    BIC = np.zeros(n_components.shape)
    AIC = np.zeros(n_components.shape)
    for i, n in enumerate(n_components):
        clf = GaussianMixture(n_components=n, covariance_type='diag', random_state=1)
        clf.fit(data)
        AIC[i] = clf.aic(data)
        BIC[i] = clf.bic(data)
    # if verbose > 0:
    #     plt.figure()
    #     plt.plot(n_components, AIC, label='AIC')
    #     plt.plot(n_components, BIC, label='BIC')
    #     plt.legend(loc=0)
    #     plt.xlabel('n_components')
    #     plt.ylabel('AIC / BIC')
    #     plt.draw()
    #     plt.waitforbuttonpress(0)  # this will wait for indefinite time
    #     plt.close()
    n_components = np.argmin(BIC)
    print('n_components', n_components)
    return n_components


def gaussianmixturemodel(img, k=8, verbose=0):
    data = img.reshape(-1, 3 if img.ndim == 3 else 1)
    # gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=1, init_params='kmeans')
    # n_components = get_n_components(data, verbose)
    n_components = 5
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=1,
                          tol=0.001, reg_covar=1e-06, max_iter=1200, n_init=1, init_params='kmeans').fit(data)
    # gmm = BayesianGaussianMixture(n_components=8, covariance_type='full',
    #                               weight_concentration_prior_type='dirichlet_distribution',
    #                               init_params="kmeans", max_iter=1200, random_state=2).fit(newdata)
    cluster = gmm.predict(data)
    cluster = cluster.reshape(img.shape[0], img.shape[1])
    cluster[cluster == 0] = cluster.max() + 1  # label problem (zeros are ignored)
    return cluster


def gaussianmixturemodel1(img):
    from sklearn.mixture import GMM
    hist, bin_edges = np.histogram(img, bins=60)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    classif = GMM(n_components=5)
    classif.fit(img.reshape((img.size, 1)))
    threshold = np.mean(classif.means_)
    binary_img = img > threshold
    plt.figure(figsize=(11, 4))
    plt.subplot(131)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(132)
    plt.plot(bin_centers, hist, lw=2)
    plt.axvline(threshold, color='r', ls='--', lw=2)
    plt.text(0.57, 0.8, 'histogram', fontsize=20, transform=plt.gca().transAxes)
    plt.yticks([])
    plt.subplot(133)
    plt.imshow(binary_img.astype(float), cmap=plt.cm.gray, interpolation='nearest')
    plt.axis('off')

    plt.subplots_adjust(wspace=0.02, hspace=0.3, top=1, bottom=0.1, left=0, right=1)
    plt.show()
