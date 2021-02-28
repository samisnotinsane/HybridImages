from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from MyConvolution import convolve
from MyHybridImages import *

def load_image_as_rgb_array(path: str) -> np.ndarray:
    image = Image.open(path)
    img = np.asarray(image)
    return img

def load_image_as_bw_array(path: str) -> np.ndarray:
    image = Image.open(path).convert('L')
    data = np.asarray(image)
    return data

def show_image(img: np.ndarray, grey: bool):
    print("Opening image...")
    if grey:
        img = np.squeeze(img)
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)
    plt.show()
    print("Image closed")

def save_image(image: np.ndarray, fname: str):
    path = 'out/' + fname + '.png'
    im = Image.fromarray(image)
    im.save(path)
    print(f"Image saved to {path}")

def print_image_meta(img: np.ndarray):
    dim = len(img.shape)
    if dim == 3:
        h,w,c = img.shape
        print(f"height: {h}, width: {w}, channel: {c}")
    elif dim == 2:
        h,w = img.shape
        print(f"height: {h}, width: {w}, channel: 1")
    print(f"Image loaded as: {type(img)}")

def my_pad(a):
    rows, cols = a.shape
    pad_rows = int(np.floor(rows/2))
    pad_cols = int(np.floor(cols/2))
    b = np.zeros((rows + 2*pad_rows, cols + 2*pad_cols))
    print(b.shape)
    print(b)

if __name__ == '__main__':
    lo_freq_img = load_image_as_rgb_array('data/dog.bmp')
    hi_freq_img = load_image_as_rgb_array('data/cat.bmp')

    lo_sigma, hi_sigma  = 3.5, 5.5 # free parameters
    hybrid = myHybridImages(lo_freq_img, lo_sigma, hi_freq_img, hi_sigma)
    save_image(hybrid, fname='cat-dog')

    show_image(hybrid, grey=False)
    print("Terminating...")
