from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from MyHybridImages import *
from MyConvolution import zero_padding

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
    # rgb_img = load_image_as_rgb_array('data/fish.bmp')
    # print_image_meta(rgb_img)
    # show_image(rgb_img, grey=False)

    bw_img = load_image_as_bw_array('data/fish.bmp')
    # print_image_meta(bw_img)
    # show_image(bw_img, grey=True)

    # print(bw_img) # pixel value at row 10, col 20
    # coordinates: origin 0,0 is at top-left corner
    # bw_img_copy = bw_img.copy()
    padded_img = zero_padding(bw_img, template=np.zeros((10, 10)))
    show_image(padded_img, grey=True)

    a = np.random.randint(low=0,high=255,size=(10,10))
    t = np.ones((6,6))
    rows, cols = t.shape
    prows = int(np.floor(rows/2))
    pcols = int(np.floor(cols/2))
    b = np.pad(a, (prows, pcols), mode='constant')

    print(bw_img.shape)
    print(padded_img.shape)
    print("Terminating...")
