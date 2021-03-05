import unittest
import numpy as np
from MyHybridImages import makeGaussianKernel

class TestMyConvolution(unittest.TestCase):

    def test_gauss_kernel(self):
        K = makeGaussianKernel(0.2)
        print(K.shape)
        self.assertNotEqual(K.shape, (0,0))

if __name__ == '__main__':
    unittest.main()