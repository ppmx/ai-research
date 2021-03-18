#!/usr/bin/env python3

""" MNIST Tensorflow """

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def show_image(img, label=None):
    """ Plots an image using pyplot """

    plt.figure()

    if label:
        plt.title(label)

    plt.imshow(img, cmap="gray")
    plt.colorbar()
    plt.grid(False)
    plt.show()

def load_image_from_file(path):
    with open(path, "rb") as reader:
        image_data = reader.read()

    img = tf.image.decode_image(image_data)
    img = tf.image.resize(img, (28, 28))
    
    # TODO: check if the image is given as grayscale
    #img = tf.image.rgb_to_grayscale(img)

    img = img / 255.0
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    parser.add_argument("filepath")
    args = parser.parse_args()

    model = tf.keras.models.load_model("./build")
    model.trainable = False

    image = load_image_from_file(args.filepath)
    
    prediction = model.predict(tf.expand_dims(image, 0))[0]
    prediction_softmax = tf.nn.softmax(prediction)

    classification = np.argmax(prediction_softmax)
    class_conf = prediction_softmax[classification] * 100

    print(title := f"Image classified as {classification} with {class_conf:02.02f}% confidence")
    for number, confidence in enumerate(prediction_softmax):
        print(f"    {number} with {confidence * 100:6.02f}% confidence")

    if args.show:
        show_image(image, title)

if __name__ == "__main__":
    main()
