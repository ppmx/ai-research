#!/usr/bin/env python3

""" Predict the content of an image using keras MobileNetV2 neural network """

#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import tensorflow as tf
import matplotlib.pyplot as plt

def img_preprocess(img):
    """ Preprocess image to prepare for prediction """

    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (224, 224))
    return tf.keras.applications.mobilenet_v2.preprocess_input(img)

def prediction_classify(prediction):
    """ The argument prediction must be basically a batch of predictions. If only
    one image was predicted, then the prediction for that single image is
    prediction[0]. top=1 sort that and than we take the first element to get the
    class with the highest score.

    Returns a 3-tuple (class-id, class-name, score).
    For example: ('n02099712', 'Labrador_retriever', 0.4181853)
    """

    decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions
    return decode_predictions(prediction, top=1)[0][0]

def classify_img(model, img):
    """ Use the model to predict and classify the image """

    prediction = model.predict(tf.expand_dims(img, 0))
    _, class_name, class_score = prediction_classify(prediction)
    return prediction, class_name, class_score

def main():
    """ Parse arguments, load model and image, predict image, show image """

    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="plot image")
    parser.add_argument("filepath", help="path to local image file")
    args = parser.parse_args()

    # load model:
    model = tf.keras.applications.MobileNetV2()
    model.trainable = False
    print("[+] model MobileNetV2 loaded")

    # open image file, decode image format and preprocess:
    img_raw = tf.image.decode_image(tf.io.read_file(args.filepath))
    img = img_preprocess(img_raw)
    print("[+] image loaded and prepared")

    # transform image to batch (it's basically [image]) and do a prediction step
    # on that image:
    _, class_name, class_score = classify_img(model, img)
    print(f"[+] image classified as '{class_name}' with {class_score*100:02.02f}% confidence")

    # show image:
    if args.show:
        plt.figure()
        plt.title(f"'{class_name}' with {class_score*100:02.02f}% confidence")
        plt.imshow(0.5 * img + 0.5)
        plt.show()


if __name__ == "__main__":
    main()
