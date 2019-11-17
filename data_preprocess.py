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

RAW_DATA_DIR = 'res/raw/test1/'

IMAGE_WIDTH: int = 205
IMAGE_HEIGHT: int = 154

should_show_first: bool = True

# If set to true will mirror the images to double the dataset
should_mirror_images: bool = True


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
    labels: List[float] = []

    # Iterate over each image in the dataset
    for img in tqdm(os.listdir(RAW_DATA_DIR)):
        try:
            img_array: np.ndarray = cv2.imread(os.path.join(RAW_DATA_DIR, img), cv2.IMREAD_COLOR)
            resized_array: np.ndarray = cv2.resize(img_array, (IMAGE_WIDTH, IMAGE_HEIGHT))
            gray_array: np.ndarray = cv2.cvtColor(resized_array, cv2.COLOR_BGR2GRAY)
            features.append(gray_array)

            # Get angle value from image filename
            angle_norm = float(search('-(.*)\\.', img).group(1))
            labels.append(angle_norm)

            # Flip image across the vertical axis
            if should_mirror_images:
                flipped_image: np.ndarray = cv2.flip(gray_array.copy(), 1)
                features.append(flipped_image)

                # Flip angle onto opposite quadrant
                labels.append(abs(1 - angle_norm))
        except cv2.error as e:
            print(e)

    # Balance data set
    right_bucket: List[Tuple[np.ndarray, float]] = []
    center_bucket: List[Tuple[np.ndarray, float]] = []
    left_bucket: List[Tuple[np.ndarray, float]] = []
    combined_bucket: List[Tuple[np.ndarray, float]] = []
    # Place images and their corresponding steering vectors into buckets for balancing
    for feature, label in zip(features, labels):
        if label < 0.4:
            right_bucket.append((feature, label))
        elif label > 0.6:
            left_bucket.append((feature, label))
        else:
            center_bucket.append((feature, label))

    print(f'Right turn quantity: {len(right_bucket)}')
    print(f'Center turn quantity: {len(center_bucket)}')
    print(f'Left turn quantity: {len(left_bucket)}')

    min_size = min(len(right_bucket), len(center_bucket), len(left_bucket))

    # Drop overrepresented straight ahead steering values
    center_bucket = center_bucket[:min_size]

    combined_bucket.extend(right_bucket)
    combined_bucket.extend(center_bucket)
    combined_bucket.extend(left_bucket)

    # Recombine balanced buckets to form final dataset
    features.clear()
    labels.clear()
    for feature, label in combined_bucket:
        features.append(feature)
        labels.append(label)

    print(f'Balanced dataset total size: {len(features)}')

    return features, labels


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
    x_train, x_valid, y_train, y_valid = train_test_split(train_features, train_angles, test_size=0.2,
                                                          shuffle=True)

    # Add empty color dimension so input shape matches what Keras expects
    x_train = np.expand_dims(x_train, axis=-1)
    x_valid = np.expand_dims(x_valid, axis=-1)

    # Normalize features array
    x_train = x_train / 255.0
    x_valid = x_valid / 255.0

    # Convert to numpy arrays with correct data type for TPU
    x_train = np.asarray(x_train, dtype='float32')
    x_valid = np.asarray(x_valid, dtype='float32')
    y_train = np.asarray(y_train, dtype='float32')
    y_valid = np.asarray(y_valid, dtype='float32')

    if should_show_first:
        image_to_show = x_train.copy()
        image_to_show = np.squeeze(image_to_show)[3]
        plt.imshow(image_to_show, cmap='gray')
        plt.show()

    print(f'Feature array shape: {x_train.shape}')

    # Save data as compressed pickle file
    pickle_out = open('res/compressed/x_train.pickle', 'wb')
    pickle.dump(x_train, pickle_out)
    pickle_out.close()

    pickle_out = open('res/compressed/y_train.pickle', 'wb')
    pickle.dump(y_train, pickle_out)
    pickle_out.close()

    pickle_out = open('res/compressed/x_valid.pickle', 'wb')
    pickle.dump(x_valid, pickle_out)
    pickle_out.close()

    pickle_out = open('res/compressed/y_valid.pickle', 'wb')
    pickle.dump(y_valid, pickle_out)
    pickle_out.close()

    print(f'Total images processed: {len(train_features)}')
    print('>>>>>>> Pre-processing complete')
