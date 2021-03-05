import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
from scipy.signal import convolve2d
from hybridimages.convolution import convolve

def test_bw_result_realistic():
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

def bcast():
    im = np.ones((254, 254))
    k = np.ones((3,5))
    res = k * im
    print(res)

def selection():
    X, Y = 254, 254
    im = np.ones((X,Y))
    out = np.zeros_like(im)
    k = np.ones((3, 5))
    for x in range(X):
        for y in range(Y):
            out[x,y] = (im[y: y+5, x: x+3] * k).sum()
    
def simple_bcast():
    np.random.seed(5)
    im = np.random.randint(25, size=(6,6))
    k = np.ones((3,5))
    print('im:')
    print(im)
    print('k:')
    print(k)
    print('t:')
    t = im[0: 0+5, 0: 0+3]
    print(t)
    print('k*t:')
    print(k * t)

if __name__ == '__main__':
    # this test harness is to investigate a bug which causes
    # 'burn' marks to appear on a hybrid image. 
    test_bw_result_realistic()
    bcast()
    selection()
    simple_bcast()