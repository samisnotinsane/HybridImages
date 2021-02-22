import numpy as np

def zero_padding(img: np.ndarray, template: np.ndarray) -> np.ndarray:
    img_copy = img.copy()
    rows, cols = template.shape
    prows = int(np.floor(rows/2))
    pcols = int(np.floor(cols/2))
    padded_img = np.pad(img_copy, (prows, pcols), mode='constant')
    return padded_img

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
    kr, kc = kernel.shape
    if kr % 3:
        raise ValueError('Kernel cannot contain an even dimension!')
    elif kc % 3:
        raise ValueError('Kernel cannot contain an even dimension!')
    print('Great!')