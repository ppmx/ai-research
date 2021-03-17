#!/usr/bin/env python3


""" This is an experiment about how the dense layer learns to
recognize a specific shape.

This script creates a dataset with objects of shape (3, 5); the
values of this object are either 0 or 1.

The objects are like a 3x5 pixel image with pixel values that are
either black (0) or white (1).

The label for one sample (one image) is either 0 (that represents
"class a") or 1 (represents "class b"). An object is classified
as class b if and only if there is a 3-pixel horizontal line in
the middle of the second row of the image.

So, for example, an object that is considered as "class b" is:
[
    [0, 0, 1, 0, 1],
    [0, 1, 1, 1, 0], # <-- here is a 3-pixel width line in the middle
    [1, 1, 0, 1, 0]
]

The model to detect that shape consists of a flatten layer and one dense layer for 2
units (class a and b). After learning, the dense layer has the following weights:

[array([[ 0.46226653, -0.18530397],
       [ 0.20274386, -0.45495248],
       [-0.13420363, -0.77918065],
       [ 0.38428843, -0.12642018],
       [ 0.49548846, -0.02059362],
       [ 0.09345558, -0.4263613 ],
       [-1.2408439 ,  1.1901827 ],
       [-1.2758697 ,  1.1517409 ],
       [-1.5112419 ,  0.9196622 ],
       [-0.10159094, -0.60419935],
       [ 0.11871985, -0.37600404],
       [ 0.47550204, -0.03576543],
       [ 0.54160804,  0.03635015],
       [ 0.50673914,  0.01468423],
       [ 0.64116037,  0.13728654]], dtype=float32), array([ 1.8214417, -1.8214418], dtype=float32)]

The weight #7, #8 and #9 is responsible for the pixels of the shape that we want to detect. And the
weights for class b at this point are actually positive numbers with high value.
"""

import tensorflow as tf
import numpy as np

def create_dataset():
    """ Creates two datasets, both consisting of the x-values (the objects) and
    y-values (the labels).
    """

    x, y = list(), list()

    decode_chunk = lambda chunk: list(map(float, list(chunk)))

    for sample_index in range(1 << 15):
        sample = [decode_chunk(format(sample_index, "015b")[i:i+5]) for i in range(0, 15, 5)]

        # label if one if the sample has a horizontal line in the middle row:
        label = 1 if sample[1][1:4] == [1, 1, 1] else 0

        x.append(sample)
        y.append(label)

    x, y = np.array(x), np.array(y)

    # split dataset into training and validation batch:
    number_train = int(len(x) * 0.9)
    return x[:number_train], y[:number_train], x[number_train:], y[number_train:]


def create_model():
    """ Create and compile a model """

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(3, 5)),
        tf.keras.layers.Dense(2)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),

        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(dtype=tf.float32)
        ]
    )

    return model

def predict(model, sample):
    """ Compute and return softmax prediction """

    prediction = model.predict(tf.expand_dims(sample, 0))
    return tf.nn.softmax(prediction[0])

def main():
    """ Create model, create dataset, train model and show weights of dense layer """

    model = create_model()
    print("[+] model created and compiled now")

    x_train, y_train, x_val, y_val = create_dataset()
    print("[+] dataset created")

    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5)
    print("[+] model trained")

    print("[+] printing now the parameters for the dense layer!")
    print(model.layers[1].get_weights())

if __name__ == "__main__":
    main()
