#!/usr/bin/env python3

""" MNIST Tensorflow """

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def load_mnist_dataset():
    """ Load, prepare and return MNIST dataset """

    mnist = tf.keras.datasets.mnist

    (imgs_train, labels_train), (imgs_verify, labels_verify) = mnist.load_data()
    return imgs_train / 255.0, labels_train, imgs_verify / 255.0, labels_verify

def bootstrap_model():
    """ Create, train and store a Tensorflow model for MNIST """

    # Load MNIST dataset:
    imgs_train, labels_train, imgs_verify, labels_verify = load_mnist_dataset()
    print(f"[+] dataset loaded: {len(labels_train)} training samples, {len(labels_verify)} validation samples")

    # Initialize model by creating neural network structure and compiling that:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(dtype=tf.float32)
        ]
    )

    print("[+] model created and compiled")

    # Train the model:
    print("[*] starting to train the model...")
    model.fit(
        x=imgs_train, y=labels_train,
        epochs=15,
        validation_data=(imgs_verify, labels_verify)
    )

    model.save("./build")
    print("[+] model saved at './build'")

if __name__ == "__main__":
    bootstrap_model()
