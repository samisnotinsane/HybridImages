import math
import numpy as np

from MyConvolution import convolve

def myHybridImages(lowImage: np.ndarray, lowSigma: float, highImage: np.ndarray, highSigma:float) -> np.ndarray:
    """
    Create hybrid images by combining a low-pass and high-pass filtered pair.

    :param lowImage: the image to low-pass filter (either greyscale shape=(rows, cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray

    :param lowSigma: the standard deviation of the Gaussian used for low-pass filtering lowImage
    :type numpy.ndarray

    :param highSigma: the standard deviation of the Gaussian used for low-pass filtering highImage before subtraction to create the high-pass filtered image
    :type float

    :returns returns the hybrid image created
        by low-pass filtering lowImage with a Gaussian of s.d. lowSigma and combining it with a high-pass image created by subtracting highImage from highImage
        convolved with a Gaussian of s.d. highSigma. The resultant image has the same size as the input images.
    :rtype numpy.ndarray
    """
    print(f'Attempting to merge {lowImage.shape} low frequency image with {highImage.shape} high frequency image')

    low_sigma_kernel = makeGaussianKernel(lowSigma)
    print(f'Applying convolution...')
    low_pass_image = convolve(lowImage, low_sigma_kernel)
    print(f'Low-pass filter successfully applied')
    
    high_sigma_kernel = makeGaussianKernel(highSigma)
    print(f'Applying convolution...')
    low_pass_of_highimage = convolve(highImage, high_sigma_kernel)

    print(f'Acquiring High-pass filtered image...')
    high_pass_img = highImage - low_pass_of_highimage

    # visualise high_pass image by adding 128 as high freq image is 0 mean with negative values
    # show_image(high_pass_img + 128, grey=False)

    hybrid_img = low_pass_image + high_pass_img
    print(f'Hybrid image computed successfully')

    return hybrid_img

def makeGaussianKernel(sigma: float) -> np.ndarray:
    """
    Use this function to create a 2D Gaussian kernel with standard deviation sigma.
    The kernel values should sum to 1.0, and the size should be floor(8*sigma+1) or 
    floor(8*sigma+1)+1 (whichever is odd).
    """
    print('Initiating Gaussian kernel with sigma:', sigma)
    size = np.floor(8 * sigma + 1).astype(int)
    
    # ensure size remains odd
    if size % 2 == 0:
        size += 1
    print(f'Gaussian kernel dimensions: {size} x {size}')

    # create range between -1 and +1
    ax = np.linspace( -(size-1) / 2, (size-1) / 2, size)
    # fill a size x size grid with above range
    x, y = np.meshgrid(ax,ax)
    
    # construct kernel using Gauss
    K = (1/(2 * math.pi * sigma**2)) * np.exp(-((x**2 + y**2)/(2 * sigma**2)))
    return K