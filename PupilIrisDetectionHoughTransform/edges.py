from skimage.filters.rank import autolevel_percentile, autolevel
import matplotlib.pyplot as plt
from skimage.morphology import disk


def local_auto_level(image):
    selem = disk(20)
    loc_autolevel = autolevel(image, selem=selem)
    loc_perc_autolevel0 = autolevel_percentile(image, selem=selem, p0=.00, p1=1.0)
    loc_perc_autolevel1 = autolevel_percentile(image, selem=selem, p0=.01, p1=.99)
    loc_perc_autolevel2 = autolevel_percentile(image, selem=selem, p0=.05, p1=.95)
    loc_perc_autolevel3 = autolevel_percentile(image, selem=selem, p0=.1, p1=.9)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10),
                             sharex=True, sharey=True)
    ax = axes.ravel()

    title_list = ['Original',
                  'auto_level',
                  'auto-level 0%',
                  'auto-level 1%',
                  'auto-level 5%',
                  'auto-level 10%']
    image_list = [image,
                  loc_autolevel,
                  loc_perc_autolevel0,
                  loc_perc_autolevel1,
                  loc_perc_autolevel2,
                  loc_perc_autolevel3]

    for i in range(0, len(image_list)):
        ax[i].imshow(image_list[i], cmap=plt.cm.gray, vmin=0, vmax=255)
        ax[i].set_title(title_list[i])
        ax[i].axis('off')
        ax[i].set_adjustable('box-forced')

    plt.tight_layout()
    plt.show()


from skimage.filters.rank import enhance_contrast_percentile


def contrast_enhance_morfological(noisy_image):
    penh = enhance_contrast_percentile(noisy_image, disk(5), p0=.1, p1=.9)
    return penh

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10),
                             sharex='row', sharey='row')
    ax = axes.ravel()

    ax[0].imshow(noisy_image, cmap=plt.cm.gray)
    ax[0].set_title('Original')

    ax[1].imshow(penh, cmap=plt.cm.gray)
    ax[1].set_title('Local percentile morphological\n contrast enhancement')

    ax[2].imshow(noisy_image[200:350, 350:450], cmap=plt.cm.gray)

    ax[3].imshow(penh[200:350, 350:450], cmap=plt.cm.gray)

    for a in ax:
        a.axis('off')
        a.set_adjustable('box-forced')

    plt.tight_layout()
    plt.show()
