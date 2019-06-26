import cv2
import random
import numpy as np
import torch


def read_image(path):
    im = cv2.imread(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def normalize(im):
    """Normalizes images with Imagenet stats."""
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    return (im - imagenet_stats[0]) / imagenet_stats[1]


def crop(im, r, c, target_r, target_c):
    return im[r:r+target_r, c:c+target_c]


def random_crop(x, r_pix=8):
    """Returns a random crop"""
    r, c, *_ = x.shape
    r, c, *_ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    return crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)


def center_crop(x, r_pix=8):
    r, c, *_ = x.shape
    c_pix = round(r_pix*c/r)
    return crop(x, r_pix, c_pix, r-2*r_pix, c-2*c_pix)


def rotate_cv(im, deg, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):
    """ Rotates an image by deg degrees"""
    r, c, *_ = im.shape
    M = cv2.getRotationMatrix2D((c/2, r/2), deg, 1)
    return cv2.warpAffine(im, M, (c, r), borderMode=mode,
                          flags=cv2.WARP_FILL_OUTLIERS+interpolation)


def save_model(m, p):
    torch.save(m.state_dict(), p)


def load_model(m, p):
    m.load_state_dict(torch.load(p))
