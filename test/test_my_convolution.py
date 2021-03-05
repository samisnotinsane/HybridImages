import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import unittest
import numpy as np
from scipy.signal import convolve2d
from MyConvolution import convolve

class TestMyConvolution(unittest.TestCase):

    def test_shape(self):
        im = np.ones((5,5))
        k = np.ones((3,3))
        conv = convolve(im, k)
        in_shape = im.shape
        out_shape = conv.shape
        np.testing.assert_equal(out_shape, in_shape)

    def test_result(self):
        im = np.ones((5,5))
        k = np.ones((3,3))
        conv = convolve(im, k)
        exp = np.array([
            [4., 6., 6., 6., 4.],
            [6., 9., 9., 9., 6.],
            [6., 9., 9., 9., 6.],
            [6., 9., 9., 9., 6.],
            [4., 6., 6., 6., 4.]
        ])
        np.testing.assert_array_equal(conv, exp)

    def test_scipy_shape(self):
        im = np.ones((5,5))
        k = np.ones((3,3))
        conv = convolve2d(im, k, mode='same')
        in_shape = im.shape
        out_shape = conv.shape
        np.testing.assert_equal(out_shape, in_shape)
    
    def test_scipy_result(self):
        im = np.ones((5,5))
        k = np.ones((3,3))
        conv = convolve2d(im, k, mode='same')
        exp = np.array([
            [4., 6., 6., 6., 4.],
            [6., 9., 9., 9., 6.],
            [6., 9., 9., 9., 6.],
            [6., 9., 9., 9., 6.],
            [4., 6., 6., 6., 4.]
        ])
        np.testing.assert_array_equal(conv, exp)

    def test_invalid_k_dim_x(self):
        im = np.ones((5,5))
        kernel = np.ones((2,3))
        np.testing.assert_raises(ValueError, convolve, im, kernel)
    
    def test_invalid_k_dim_y(self):
        im = np.ones((5,5))
        kernel = np.ones((3,2))
        np.testing.assert_raises(ValueError, convolve, im, kernel)

    def test_zero_padding(self):
        im = np.ones((5,5))
        k = np.ones((3,3))
        conv = convolve(im, k)
        corners = conv[[0, 0, -1, -1], [0, -1, -1, 0]]
        exp = np.array([4., 4., 4., 4.])
        np.testing.assert_array_equal(corners, exp)
    
    def test_colour_shape(self):
        im = np.ones((5,5,3))
        k = np.ones((3,3))
        conv = convolve(im, k)
        in_shape = im.shape
        out_shape = conv.shape
        np.testing.assert_equal(out_shape, in_shape)
    
    def test_colour_shape_scipy(self):
        im = np.ones((5,5,3))
        k = np.ones((3,3))
        for i in range(3):
            im_d = im[:, :, i]
            conv = convolve2d(im[:, :, i], k, mode='same', boundary='fill', fillvalue=0)
            np.testing.assert_equal(conv.shape, im_d.shape)

    def test_colour_result(self):
        im = np.ones((5,5,3))
        k = np.ones((3,3))
        for i in range(3):
            scipy_conv_d = convolve2d(im[:, :, i], k, mode='same', boundary='fill', fillvalue=0)
            conv_d = convolve(im[:, :, i], k)
            np.testing.assert_equal(conv_d, scipy_conv_d)

    def test_bw_result_realistic(self):
        np.random.seed(70)
        x_pixels = np.random.randint(225, 265)
        y_pixels = np.random.randint(225, 265)
        max_k_xy = 32
        k_sample_space = np.arange(1, max_k_xy, 2)
        k_x = np.random.choice(k_sample_space)
        k_y = np.random.choice(k_sample_space)
        im = np.ones((x_pixels, y_pixels))
        k = np.ones((k_x, k_y))
        scipy_conv = convolve2d(im, k, mode='same', boundary='fill', fillvalue=0)
        conv = convolve(im, k)
        np.testing.assert_equal(conv, scipy_conv)

    def test_colour_result_realistic(self):
        np.random.seed(68)
        x_pixels = np.random.randint(225, 265)
        y_pixels = np.random.randint(225, 265)
        max_k_xy = 32
        k_sample_space = np.arange(1, max_k_xy, 2)
        k_x = np.random.choice(k_sample_space)
        k_y = np.random.choice(k_sample_space)
        im = np.ones((x_pixels, y_pixels, 3))
        k = np.ones((k_x, k_y))
        for i in range(3):
            scipy_conv_d = convolve2d(im[:, :, i], k, mode='same', boundary='fill', fillvalue=0)
            conv_d = convolve(im[:, :, i], k)
            np.testing.assert_equal(conv_d, scipy_conv_d)

if __name__ == '__main__':
    unittest.main()