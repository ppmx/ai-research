#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import predict
import fgsm

# load model:
model = tf.keras.applications.MobileNetV2()
model.trainable = False
print("[+] model loaded")

# download or resure image file, open and decode image format, preprocess:
img_name = "YellowLabradorLooking_new.jpg"
img_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg"

img_path = tf.keras.utils.get_file(img_name, img_url)
img_raw = tf.image.decode_image(tf.io.read_file(img_path))
img = predict.img_preprocess(img_raw)
print("[+] image loaded and prepared")

# transform image to batch (it's basically [image]) and do a prediction step
# on that image:
prediction, class_name, class_score = predict.classify_img(model, img)
print(f"[+] original image classified as '{class_name}' with {class_score*100:02.02f}% confidence")

# Apply Fast-Gradient-Sign-Method: gradient descent (impersonation) on class target_index
num_steps, eps, target_index = 20, -0.01, 1

target_shape = prediction.shape[-1] # which is 1000 for MobileNetV2
target = tf.one_hot(target_index, target_shape)
target = tf.reshape(target, (1, target_shape))

fgsm_env = fgsm.FGSM(model, tf.keras.losses.CategoricalCrossentropy())
adversarial_example = fgsm_env.attack(img, target, eps, num_steps, predict.classify_img)

# show adversarial example:
plt.figure()
plt.imshow(0.5 * adversarial_example + 0.5)
plt.show()
