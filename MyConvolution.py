import numpy as np

def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve an image with a kernel assuming zero-padding of the image to handle the borders

    :param image: the image (either greyscale shape=(rows, cols) or colour shape=(rows, cols, channels))
    :type numpy.ndarray

    :param: kernel: the kernel (shape=(kheight, kwidth); both dimensions odd)
    :type: numpy.ndarray

    :returns the convolved image (of the same shape as the input image)
    :rtype numpy.ndarray
    """
    im_count_row, im_count_col = image.shape
    kern_count_row, kern_count_col = kernel.shape
    if (kern_count_row % 2) == 0:
        raise ValueError('Kernel cannot be an even dimension!')
    if (kern_count_col % 2) == 0:
        raise ValueError('Kernel cannot be an even dimension!')
    out = np.zeros_like(image)
    # width of padding is half the size of kernel
    pad_count_row = np.floor((kern_count_row/2)).astype(int)
    pad_count_col = np.floor((kern_count_col/2)).astype(int)
    image_padded = np.pad(image, (pad_count_row, pad_count_col), mode='constant')
    for x in range(im_count_col):
        for y in range(im_count_row):
            # guard against going past padding
            if x >= (im_count_col - kern_count_col):
                continue
            if y >= (im_count_row - kern_count_row):
                continue
            # convolve
            image_section = image_padded[x: x+kern_count_col, y: y+kern_count_row]
            out[x,y] = (kernel * image_section).sum()
    return out
