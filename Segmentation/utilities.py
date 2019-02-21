import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from glob import glob
import seaborn as sns
import pickle
import time

def show(mat):
    plt.figure(figsize=(10,10))
    plt.imshow(mat,cmap='gray')
    # Remove xticks, yticks
    plt.xticks([])
    plt.yticks([])
    plt.show()


def fix_json_fname(fname):
    """Add JSON suffix to file name if it's missing"""
    if fname.lower().endswith('.json'):
        return fname
    else:
        return fname + ".json"


def count_combs(lst):
    """Receives a list of lists. Returns the number of combinations."""
    num_combs = 1
    for mem_len in (len(list(x)) for x in lst):
        num_combs *= mem_len
    return num_combs


def save_img(img, fname):
    if len(img.shape) != 3:
        RGB_img = np.zeros([512,512,3], dtype='uint8')
        # keep the same image in all 3 colors
        for i in range(3):
            RGB_img[:,:,i] = np.ndarray.astype(norm_data(img)*255,'uint8')
        img = RGB_img
    im = Image.fromarray(img)
    im.save(f"{fname}.jpeg")


def norm_data(data):
    """Normalize an image"""
    data -= np.min(data)  # remove negative values
    data /= np.max(data)  # normalize image to 0-1
    return data


def pickle_object(obj, filename):
    if not filename.endswith('pickle'):
        filename = filename + ".pickle"

    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        print("Results pickled")