"""Processes data to an appropriate format before training."""
import os
import pickle
from re import search
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

RAW_DATA_DIR = '../res/training_data/raw/'

IMAGE_WIDTH: int = 820
IMAGE_HEIGHT: int = 616

should_show_first: bool = True

# If set to true will mirror the images to double the dataset
should_mirror_images: bool = False


def process_image_for_model(img_to_process: np.ndarray) -> np.ndarray:
    """
    Resize an image before using it for training.
    :param img_to_process: image to process
    :return: processed image
    """
    # Resize the array to the size standard
    resized_array: np.ndarray = cv2.resize(img_to_process, (IMAGE_WIDTH, IMAGE_HEIGHT))

    # Normalize the pixel values
    resized_array = resized_array / 255.0

    # Expand the dimensions out to what TensorFlow is expecting
    resized_array = np.expand_dims(resized_array, axis=2)
    resized_array = np.expand_dims(resized_array, axis=0)

    return resized_array


def create_training_data() -> Tuple[List[np.ndarray], List[float]]:
    """
    Create a list of images that will be used as training data.
    :return: list of images
    """
    features: List[np.ndarray] = []
    angles: List[float] = []

    # Iterate over each image in the dataset
    for img in tqdm(os.listdir(RAW_DATA_DIR)):
        try:
            img_array: np.ndarray = cv2.imread(os.path.join(RAW_DATA_DIR, img), cv2.IMREAD_GRAYSCALE)
            resized_array: np.ndarray = cv2.resize(img_array, (IMAGE_WIDTH, IMAGE_HEIGHT))
            features.append(resized_array)

            # Get angle value from image filename
            angle_rad: float = float(search('-(.*)\\.', img).group(1))
            angles.append(angle_rad)

            # Flip image across the vertical axis
            if should_mirror_images:
                flipped_image: np.ndarray = cv2.flip(resized_array.copy(), 1)
                features.append(flipped_image)

                # Flip angle onto opposite quadrant
                if angle_rad > np.radians(90):  # Change Quadrant II angle to Quadrant I angle
                    angle_rad = np.absolute(angle_rad - np.pi)
                elif np.radians(90) > angle_rad:  # Change Quadrant I angle to Quadrant II angle
                    angle_rad = np.pi - angle_rad
                angles.append(angle_rad)
        except cv2.error as e:
            print(e)

    return features, angles


def test_prediction():
    """Tests a single prediction the regression model made."""
    image_number = 120
    model_name = 'optimized_regression.model'
    regression_model: tf.keras.Model = tf.keras.load_model(f'../models/steering_vector/{model_name}')
    file_image = cv2.imread(os.path.join(RAW_DATA_DIR, f'{image_number}.png'), cv2.IMREAD_GRAYSCALE)
    processed_image = process_image_for_model(file_image)

    print(f'Processed image shape: {processed_image.shape}')
    img_to_show = np.squeeze(processed_image.copy())
    plt.imshow(img_to_show, cmap='gray')
    plt.show()

    prediction = regression_model.predict([processed_image])
    confidence_score = np.amax(prediction[0])
    print(f'Confidence Score: {confidence_score}')


if __name__ == '__main__':
    train_features, train_angles = create_training_data()

    # Split the data and shuffle it to avoid over-fitting one particular class
    x_train, x_valid, y_train, y_valid = train_test_split(train_features, train_angles, test_size=0.3,
                                                          shuffle=True)

    x_train = np.asarray(x_train)
    x_valid = np.asarray(x_valid)

    # Normalize features array
    x_train = x_train / 255.0
    x_valid = x_valid / 255.0

    # Convert labels to one-hot array
    y_train = tf.keras.to_categorical(y_train)
    y_valid = tf.keras.to_categorical(y_valid)

    if should_show_first:
        image_to_show = x_train.copy()
        image_to_show = np.squeeze(image_to_show)[3]
        plt.imshow(image_to_show, cmap='gray')
        plt.show()

    x_train = np.expand_dims(x_train, axis=3)
    x_valid = np.expand_dims(x_valid, axis=3)
    print(f'Feature array shape: {x_train.shape}')

    # Save data as compressed pickle file
    pickle_out = open('../res/training_data/compressed/x_train.pickle', 'wb')
    pickle.dump(x_train, pickle_out)
    pickle_out.close()

    pickle_out = open('../res/training_data/compressed/y_train.pickle', 'wb')
    pickle.dump(y_train, pickle_out)
    pickle_out.close()

    pickle_out = open('../res/training_data/compressed/x_valid.pickle', 'wb')
    pickle.dump(x_valid, pickle_out)
    pickle_out.close()

    pickle_out = open('../res/training_data/compressed/y_valid.pickle', 'wb')
    pickle.dump(y_valid, pickle_out)
    pickle_out.close()

    print(f'Total images processed: {len(train_features)}')
    print('>>>>>>> Pre-processing complete')
