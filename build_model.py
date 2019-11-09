import os
import pickle

import tensorflow as tf

# Support running in a Colab TPU accelerated runtime if available
TPU_MODE = True if os.environ.get('COLAB_TPU_ADDR') else False

if TPU_MODE:
    PICKLE_DATA_DIR = '/content/'
else:
    PICKLE_DATA_DIR = '../res/TrainingData/compressed/'

batch_size = 39
num_epochs = 8

print('>  ====== Loading compressed training data')

# Load processed compressed data from disk
pickle_in = open(PICKLE_DATA_DIR + 'x_train.pickle', 'rb')
x_train = pickle.load(pickle_in)

pickle_in = open(PICKLE_DATA_DIR + 'y_train.pickle', 'rb')
y_train = pickle.load(pickle_in)

pickle_in = open(PICKLE_DATA_DIR + 'x_valid.pickle', 'rb')
x_valid = pickle.load(pickle_in)

pickle_in = open(PICKLE_DATA_DIR + 'y_valid.pickle', 'rb')
y_valid = pickle.load(pickle_in)

print('>  ====== Compressed training data loaded')

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.BatchNormalization(input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='elu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.1))

model.add(tf.keras.layers.BatchNormalization(input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='elu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.1))

model.add(tf.keras.layers.BatchNormalization(input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='elu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.1))

model.add(tf.keras.layers.Flatten())  # Convert the 3D feature maps to 1D feature vectors
model.add(tf.keras.layers.Dense(32))
model.add(tf.keras.layers.Activation('elu'))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(1))
model.add(tf.keras.layers.Activation('softmax'))

if TPU_MODE:
    tpu_model = tf.contrib.tpu.keras_to_tpu_model(
        model,
        strategy=tf.contrib.tpu.TPUDistributionStrategy(
            tf.contrib.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
        )
    )

    tpu_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-3),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=['accuracy']
    )

    tpu_model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(x_valid, y_valid))

    cpu_model = tpu_model.sync_to_cpu()
    cpu_model.save('optimized_classification_graph.model')
else:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-3),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=['accuracy']
    )
    model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(x_valid, y_valid))
    model.save('optimized_classification_graph.model')

print('>>>>>>> Training complete')
