import os
import pickle
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split

IMG_WIDTH = 100
IMG_HEIGHT = 120

should_show_first = True


def process_image_for_model(img_to_process: np.ndarray):
    # resize the array to the size standard
    resized_array = cv2.resize(img_to_process, (IMG_WIDTH, IMG_HEIGHT))
    # Normalize the pixel values
    resized_array = resized_array / 255.0
    # Expand the dimensions out to what TensorFlow is expecting
    resized_array = np.expand_dims(resized_array, axis=2)
    resized_array = np.expand_dims(resized_array, axis=0)
    return resized_array
