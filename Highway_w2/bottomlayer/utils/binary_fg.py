import numpy as np
import scipy.ndimage as nimg
from skimage.morphology import convex_hull_object
import time
import cv2

def dog(in_img, param1, param2):
    """
    calculating dog of the image, and the param  of two kernel
    """
    #g1_img = nimg.gaussian_filter(in_img, param1)
    #g2_img = nimg.gaussian_filter(in_img, param2)
    g1_img = cv2.GaussianBlur(in_img, (0, 0), param1)
    g2_img = cv2.GaussianBlur(in_img, (0, 0), param2)

    out_img = g1_img.astype('int16') - g2_img.astype('int16')
    out_img = np.clip(out_img, 0, 255)
    return out_img.astype('uint8')

def thresh(in_img, val):
    """
    threshold the in_image (gray) by val, range in [0, 255]
    pixel = (pixel>val)?1:0
    """
    mask = in_img>val
    out_img = in_img *  mask
    return (out_img>0).astype('uint8')*255

def dilation(in_img, w, h):
    """
    dilation the in_img with kernel h*w
    """
    strc = np.ones((h, w), dtype=np.uint8)
    out_img = cv2.dilate(in_img, strc, iterations=1)
    return out_img

def convex(in_img):
    in_img[convex_hull_object(in_img)] = 255
    return in_img

def run(in_img):
    """
    the main run function
    """
    #0.09s
    initt = time.time()
    dog_img = dog(in_img, 1, 15)

    #0.025s
    dog_img = cv2.medianBlur(dog_img, 3)

    #0.002s
    bin_img = thresh(dog_img, 20)

    #0.016s
    median_img = cv2.medianBlur(bin_img, 3)

    #0.018s
    dilation_img = dilation(median_img, 10, 10)

    out_img = dilation_img
    out_img = thresh(out_img, 10)
    return out_img




if  __name__=='__main__':
    img = cv2.imread('../images/0f60cd6d-5362-4dda-9b83-4e37fb7b54da.png')
    img = np.max(img, axis=-1)
    out = run(img)
    cv2.imshow('img', img)
    cv2.imshow('out', out)
    cv2.waitKey(0)


