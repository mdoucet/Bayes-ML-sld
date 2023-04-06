import sys
import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from reflectivity_model import calculate_reflectivity_from_profile
import warnings
warnings.filterwarnings('ignore', module='numpy')
warnings.filterwarnings('ignore')


class VAELoss(keras.losses.Loss):
    # We probably need to pass the reflectivity model to __init__
    def __init__(self, name='reflectivity', kl_weight=2, reco_weight=200,
                 z_left=-100, z_right=900, dz=10):
        super().__init__(name=name)
        self.kl_weight = kl_weight
        self.reco_weight = reco_weight
        self.z_left = z_left
        self.z_right = z_right
        self.dz = dz

    def call(self, y_true, y_pred):
        # The predictions from the encoder are concatenated, split them
        z_mean, z_log_var, z = tf.split(y_pred, 3, axis=1)

        # Monotonicity condition
        #mono_loss = tf.square((z[1:] - z[:-1])/z[1:] )

        # Reconstruction loss (MSE)
        steps = np.arange(101., dtype='float32') * self.dz
        #steps = np.arange(y_true.shape[1]) * self.dz
        q = np.logspace(np.log10(0.009), np.log10(0.16), num=50)

        r_true = calculate_reflectivity_from_profile(q, steps, y_true[1].numpy())
        r_pred = calculate_reflectivity_from_profile(q, steps, z_mean[1].numpy())

        reconstruction_loss = tf.square(r_pred - r_true)
        reconstruction_loss = tf.reduce_mean(reconstruction_loss)
        reconstruction_loss = tf.cast(reconstruction_loss, tf.float32)

        nll_loss = 0.5 * ( np.log(2*np.pi) + z_log_var + tf.square(z-y_true)/tf.exp(z_log_var) )
        nll_loss = tf.abs(nll_loss)
        nll_loss = tf.reduce_mean(nll_loss)

        kl_loss = -0.5 * (z_log_var - tf.square(z_mean-y_true) - tf.exp(z_log_var) + 1)
        #kl_loss = -0.5 * (- tf.square(z_mean-y_true) - tf.exp(z_log_var) + 1)
        kl_loss = tf.abs(kl_loss)
        kl_loss = tf.reduce_mean(kl_loss)
        #return self.kl_weight * kl_loss + nll_loss
        return self.kl_weight * kl_loss + nll_loss + self.reco_weight * reconstruction_loss


def reconstruction_mse_metric(y_true, y_pred, reco_weight=200):
    # The predictions from the encoder are concatenated, split them
    z_mean, z_log_var, z = tf.split(y_pred, 3, axis=1)

    steps = np.arange(y_true.shape[1]) * 10
    q = np.logspace(np.log10(0.009), np.log10(0.16), num=50)

    r_true = calculate_reflectivity_from_profile(q, steps, y_true[1].numpy())
    r_pred = calculate_reflectivity_from_profile(q, steps, z_mean[1].numpy())

    reconstruction_loss = tf.square(r_pred**2 - r_true**2)
    #reconstruction_loss = tf.reduce_mean(reconstruction_loss)
    return reco_weight * tf.reduce_mean(reconstruction_loss)


def nll_metric(y_true, y_pred):
    # The predictions from the encoder are concatenated, split them
    z_mean, z_log_var, z = tf.split(y_pred, 3, axis=1)

    loss = 0.5 * ( np.log(2*np.pi) + z_log_var + tf.square(z-y_true)/tf.exp(z_log_var) )
    return tf.reduce_mean(loss)


def mse_metric(y_true, y_pred):
    # The predictions from the encoder are concatenated, split them
    z_mean, z_log_var, z = tf.split(y_pred, 3, axis=1)
    loss = tf.square(z-y_true)
    return loss


def mono_metric(y_true, y_pred):
    # The predictions from the encoder are concatenated, split them
    z_mean, z_log_var, z = tf.split(y_pred, 3, axis=1)
    loss = tf.square((z_mean[1:] - z_mean[:-1])/tf.exp(z_log_var[1:])/100)
    return loss


def kl_metric(y_true, y_pred, kl_weight=2):
    # The predictions from the encoder are concatenated, split them
    z_mean, z_log_var, z = tf.split(y_pred, 3, axis=1)

    kl_loss = -0.5 * (z_log_var - tf.square(z_mean-y_true) - tf.exp(z_log_var) + 1)
    return kl_weight * tf.reduce_mean(kl_loss)
