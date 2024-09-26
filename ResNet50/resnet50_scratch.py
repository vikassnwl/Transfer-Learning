import tensorflow as tf
import numpy as np


def residual_block(x, filters, strides=1, conv_shortcut=False):
    shortcut = x
    x = tf.keras.layers.Conv2D(filters, 1, strides=strides, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(filters, 3, strides=1, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(filters*4, 1, strides=1, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if conv_shortcut:
        shortcut = tf.keras.layers.Conv2D(filters*4, 1, strides=strides, padding="same")(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Activation("relu")(x)
    return x


def build_resnet50(input_shape, classes):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, 7, strides=2, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

    x = residual_block(x, 64, 1, conv_shortcut=True)
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = residual_block(x, 128, 2, conv_shortcut=True)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = residual_block(x, 128)

    x = residual_block(x, 256, 2, conv_shortcut=True)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = residual_block(x, 256)

    x = residual_block(x, 512, 2, conv_shortcut=True)
    x = residual_block(x, 512)
    x = residual_block(x, 512)

    x = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(classes, activation="softmax")(x)

    model = tf.keras.models.Model(inputs, outputs)
    return model