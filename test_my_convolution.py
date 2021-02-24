import unittest
import numpy as np
from MyConvolution import convolve

class TestMyConvolution(unittest.TestCase):

    # no edge case: 50x50 image with 3x3 kernel
    def test_averaging_syn(self):
        np.random.seed(0)
        image = np.random.randint(low=0, high=255, size=(22,6))
        kernel = (1/9) * np.ones((3,3))
        conv_img = convolve(image, kernel)
        print(f"Image:\n{image}\nKernel:\n{kernel}\nConvolved Image:\n{conv_img}")
        self.assertTrue(np.array_equal(image.shape, conv_img.shape))

    def test_even_row_kernel_dim(self):
        image = np.random.randint(low=0,high=255,size=(10,10))
        kernel = (1/9) * np.ones((2,3))
        with self.assertRaises(ValueError):
            convolve(image, kernel)
    
    def test_even_col_kernel_dim(self):
        image = np.random.randint(low=0,high=255,size=(10,10))
        kernel = (1/9) * np.ones((3,2))
        with self.assertRaises(ValueError):
            convolve(image, kernel)

if __name__ == '__main__':
    unittest.main()