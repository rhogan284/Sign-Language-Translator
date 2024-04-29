import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder

gloss_count = 100

train_data = np.load('../Arrays/1000_landmarks_train.npy')
train_labels = np.load('../Arrays/1000_labels_train.npy')
test_data = np.load('../Arrays/1000_landmarks_test.npy')
test_labels = np.load('../Arrays/1000_labels_test.npy')

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
test_labels = label_encoder.transform(test_labels)


def inception_module(input_tensor, filter_channels):
    conv1x1 = layers.Conv2D(filter_channels, (1, 1), padding='same', activation='relu')(input_tensor)

    conv3x3 = layers.Conv2D(filter_channels, (3, 3), padding='same', activation='relu')(input_tensor)

    conv5x5 = layers.Conv2D(filter_channels, (5, 5), padding='same', activation='relu')(input_tensor)

    max_pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_tensor)
    max_pool = layers.Conv2D(filter_channels, (1, 1), padding='same', activation='relu')(max_pool)

    output = layers.concatenate([conv1x1, conv3x3, conv5x5, max_pool], axis=-1)
    output = layers.Dropout(0.5)(output)

    return output


def build_hpe_vgg_incep(input_shape=(262, 21, 3)):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = inception_module(x, filter_channels=32)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(100, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    return model


model = build_hpe_vgg_incep()

model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

history = model.fit(train_data, train_labels, epochs=50,
                    validation_data=(test_data, test_labels))
