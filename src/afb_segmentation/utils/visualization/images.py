"""Display source images and masks side by side."""

import matplotlib.pyplot as plt


def show_images_side_by_side(
    img1,
    img2,
    title1="Original Image",
    title2="Mask",
    cmap1="gray",
    cmap2="magma",
    figsize=(10, 5),
    dpi=300,
):
    """Display two images side by side in a publication-quality figure.

    Parameters
    ----------
    img1 : PIL.Image or numpy.ndarray
        First image, such as the original image.
    img2 : PIL.Image or numpy.ndarray
        Second image, such as a mask or segmentation result.
    title1 : str
        Title for the first image.
    title2 : str
        Title for the second image.
    cmap1 : str
        Colormap for the first image (default 'gray').
    cmap2 : str
        Colormap for the second image (default 'magma').
    figsize : tuple, default=(10, 5)
        Figure size in inches.
    dpi : int, default=300
        Figure resolution in dots per inch.

    Returns
    -------
    None
        Displays the figure with ``matplotlib.pyplot.show``.
    """

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("white")

    axes[0].imshow(img1, cmap=cmap1)
    axes[0].set_title(title1, fontsize=12, fontweight="semibold", pad=10)

    axes[1].imshow(img2, cmap=cmap2)
    axes[1].set_title(title2, fontsize=12, fontweight="semibold", pad=10)

    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.tight_layout(pad=1.0)
    plt.show()
