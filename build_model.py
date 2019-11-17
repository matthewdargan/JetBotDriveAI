import os
import pickle

import numpy as np
import tensorflow as tf

# Batch size should be a divisor of number of training samples
batch_size = 181
num_epochs = 8

print(f'Running on TensorFlow v.{tf.__version__} with Numpy v.{np.__version__}')

print('>  ====== Loading compressed training data')

# Load processed compressed data from disk
pickle_in = open('x_train.pickle', 'rb')
x_train = pickle.load(pickle_in)

pickle_in = open('y_train.pickle', 'rb')
y_train = pickle.load(pickle_in)

pickle_in = open('x_valid.pickle', 'rb')
x_valid = pickle.load(pickle_in)

pickle_in = open('y_valid.pickle', 'rb')
y_valid = pickle.load(pickle_in)

print('>  ====== Compressed training data loaded')


def define_model():
    model = tf.keras.models.Sequential()

    # Convolutional feature map layers
    model.add(tf.keras.layers.BatchNormalization(input_shape=x_train.shape[1:]))
    model.add(tf.keras.layers.Conv2D(24, (5, 5), padding='same', activation='elu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.1))

    model.add(tf.keras.layers.BatchNormalization(input_shape=x_train.shape[1:]))
    model.add(tf.keras.layers.Conv2D(36, (5, 5), padding='same', activation='elu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.1))

    model.add(tf.keras.layers.BatchNormalization(input_shape=x_train.shape[1:]))
    model.add(tf.keras.layers.Conv2D(48, (5, 5), padding='same', activation='elu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.1))

    model.add(tf.keras.layers.BatchNormalization(input_shape=x_train.shape[1:]))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='elu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.1))

    model.add(tf.keras.layers.BatchNormalization(input_shape=x_train.shape[1:]))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='elu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.1))

    # Convert the 3D feature maps to 1D feature vectors
    model.add(tf.keras.layers.Flatten())

    # Fully-connected layers
    model.add(tf.keras.layers.Dense(100))
    model.add(tf.keras.layers.Activation('elu'))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(50))
    model.add(tf.keras.layers.Activation('elu'))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Activation('elu'))
    model.add(tf.keras.layers.Dropout(0.2))

    # Output layer: normalized steering angle
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation('softmax'))

    return model


# Obtain a resolver and connect to a TPU runtime
resolver = tf.contrib.cluster_resolver.TPUClusterResolver('grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.contrib.distribute.initialize_tpu_system(resolver)
strategy = tf.contrib.distribute.TPUStrategy(resolver)

# Define and compile the model
with strategy.scope():
    model = define_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-3),
        loss='mse',
        metrics=['mse', 'mae']
    )

# Train the model
model.fit(
    x_train.astype(np.float32),
    y_train.astype(np.float32),
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=(x_valid.astype(np.float32), y_valid.astype(np.float32))
)

model.save_weights('jetCNN.h5', overwrite=True)

print('>>>>>>> Training complete')
