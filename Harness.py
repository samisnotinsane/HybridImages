from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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

def print_image_meta(img: np.ndarray):
    dim = len(img.shape)
    if dim == 3:
        h,w,c = img.shape
        print(f"height: {h}, width: {w}, channel: {c}")
    elif dim == 2:
        h,w = img.shape
        print(f"height: {h}, width: {w}, channel: 1")
    print(f"Image loaded as: {type(img)}")
    

if __name__ == '__main__':
    rgb_img = load_image_as_rgb_array('data/fish.bmp')
    print_image_meta(rgb_img)
    show_image(rgb_img, grey=False)

    bw_img = load_image_as_bw_array('data/fish.bmp')
    print_image_meta(bw_img)
    show_image(bw_img, grey=True)

    print("Terminating...")
