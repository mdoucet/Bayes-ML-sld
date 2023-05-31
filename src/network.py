import os
import tensorflow as tf
from tensorflow import keras

import pandas as pd

from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform

from loss import ReconstructionLoss, kl_metric, nll_metric, reconstruction_mse_metric, mse_metric, mono_metric


class SamplingLayer(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(z_log_var) * epsilon


class VariationalModel(keras.Model):

    def __init__(self, input_dim=150, latent_dim=421, kl_weight=1, name="sld_vae", **kwargs):
        super(VariationalModel, self).__init__(name=name, **kwargs)
        self.encoder = create_encoder(input_dim, latent_dim)

    def call(self, inputs):
        z_mean, z_log_var = tf.split(self.encoder(inputs), 2, axis=1)
        # We could move the sampling to the loss function and avoid
        # having to define a model class.
        z = SamplingLayer()([z_mean, z_log_var])
        # Here we should add reconstruction loss, unless we take care of it in the loss function
        # self.add_loss(self.kl_weight * kl_divergence)
        return keras.layers.concatenate([z_mean, z_log_var, z])


def variational_model(train_data, train_pars, dz=10, qmax=0.16):
    model = VariationalModel(train_data.shape[1], train_pars.shape[1])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    model.compile(optimizer, run_eagerly=True,
                  loss=ReconstructionLoss(kl_weight=1, dz=dz, qmax=qmax),
                  #metrics=[kl_metric, nll_metric, reconstruction_mse_metric]
                  )
    return model


def create_encoder(input_dim, latent_dim, latent_dim_multiplier=2):
    model = keras.models.Sequential([keras.Input(shape=(input_dim, 1)),
                                     keras.layers.Conv1D(filters=40, kernel_size=5, strides=1,
                                                         activation='relu', padding='same'),
                                     keras.layers.MaxPool1D(pool_size=2),
                                     keras.layers.Conv1D(filters=40, kernel_size=5, strides=1, activation='relu', padding='same'),
                                     keras.layers.MaxPool1D(pool_size=2),
                                     keras.layers.Conv1D(filters=50, kernel_size=5, strides=1, activation='relu', padding='same'),
                                     # Added conv1d
                                     keras.layers.MaxPool1D(pool_size=2),
                                     keras.layers.Conv1D(filters=50, kernel_size=5, strides=1, activation='relu', padding='same'),
                                     keras.layers.Flatten(),
                                     keras.layers.Dense(400, activation='relu'),
                                     # Two new layers not in v1
                                     keras.layers.Dense(400, activation='relu'),
                                     keras.layers.Dense(200, activation='relu'),
                                     keras.layers.Dense(latent_dim*latent_dim_multiplier),
                                     ])
    return model


def save_model(model, model_name, data_dir=''):
    """
        Save a trained model
        @param model: TensorFlow model
        @param model_name: base name for saved files
        @param data_dir: output directory
    """
    # serialize model to JSON
    model_json = model.encoder.to_json()
    with open(os.path.join(data_dir, "%s.json" % model_name), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.encoder.save_weights(os.path.join(data_dir, "%s.h5" % model_name))


def load_model(model_name, data_dir=''):
    """
        Load and return a trained model
        @param model_name: base name for saved files
        @param data_dir: directory containing trained model
    """
    # load json and create model
    json_file = open(os.path.join(data_dir, '%s.json' % model_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    #with CustomObjectScope({'GlorotUniform': glorot_uniform(), 'SamplingLayer': SamplingLayer}):
    encoder = model_from_json(loaded_model_json)
    input_dim = encoder.input_spec[0].shape[1]
    output_dim = encoder.output.type_spec.shape[1]
    print("Dimensions: %s %s" % (input_dim, output_dim))
    # load weights into new model
    encoder.load_weights(os.path.join(data_dir, '%s.h5' % model_name))
    model = VariationalModel(input_dim=input_dim, latent_dim=output_dim)
    model.encoder = encoder
    return model
