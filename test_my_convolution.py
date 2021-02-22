import unittest
import numpy as np
from MyConvolution import convolve

class TestMyConvolution(unittest.TestCase):

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