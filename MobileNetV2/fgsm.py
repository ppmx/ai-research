#!/usr/bin/env python3

import itertools
import tensorflow as tf

class FGSM:
    def __init__(self, model, loss_function):
        self.model = model
        self.loss_f = loss_function

    def compute_gradient(self, img, target):
        batch = tf.expand_dims(img, 0)

        with tf.GradientTape() as tape:
            tape.watch(batch)

            prediction = self.model(batch)
            loss = self.loss_f(target, prediction)

        return tape.gradient(loss, batch)

    def create_pattern(self, img, target):
        gradient = self.compute_gradient(img, target)
        return tf.sign(gradient)[0]

    def attack(self, img, target, epsilon, num_steps, classify_f=None):
        """ if verbose is true then the classify_f 
        """

        if isinstance(epsilon, float):
            epsilon = itertools.repeat(epsilon)
        else:
            assert len(epsilon) >= num_steps

        adv_img = img

        for i, round_epsilon in zip(range(num_steps), epsilon):
            perturbation = self.create_pattern(adv_img, target)
            adv_img = adv_img + round_epsilon * perturbation
            adv_img = tf.clip_by_value(adv_img, -1, 1)

            if classify_f:
                _, class_name, class_score = classify_f(self.model, adv_img)
                print(f"[*] fgsm iteration {i+1}: {class_name} with {class_score*100:02.02f}")

        return adv_img
