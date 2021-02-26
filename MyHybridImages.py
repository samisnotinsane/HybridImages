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

    return np.zeros((1,1))

def makeGaussianKernel(sigma: float) -> np.ndarray:
    """
    Use this function to create a 2D Gaussian kernel with standard deviation sigma.
    The kernel values should sum to 1.0, and the size should be floor(8*sigma+1) or 
    floor(8*sigma+1)+1 (whichever is odd).
    """
    size = np.floor(8 * sigma + 1).astype(int)
    # ensure size remains odd
    if size % 2 == 0:
        size += 1
    mean = np.array([0,0])
    cov = np.array(
        [[1,0],
         [0,1]]
    )
    rng = np.random.default_rng()
    K = rng.multivariate_normal(mean, cov, (size,size))
    return K