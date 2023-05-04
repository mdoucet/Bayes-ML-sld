import sys
import os
import numpy as np
np.random.seed(42)
import argparse
import time

import tensorflow as tf
assert tf.multiply(6, 7).numpy() == 42

from tensorflow import keras
import pandas as pd

import json
import refl1d
from refl1d.names import *

from matplotlib import pyplot as plt
import matplotlib.lines as mlines



import warnings
warnings.filterwarnings('ignore', module='numpy')
warnings.filterwarnings('ignore')

nval=50000

print(tf.__version__)

def workflow(config, n_train=None, v_size=None,
             create=False, use_errors=False):
    """
        Overall workflow of the training process

        @param v_size: size of the validation set
    """
    training_dir = config['train_dir']
    model = config['model']
    parameters = config['parameters']
    epoch = config['epoch']
    name = config['name']
    print("Processing %s" % name)
    print("TF version: %s" % tf.__version__)

    q_ref = None

    # Define the reflectivity model we are going to be generating/using
    m = reflectivity_model.ReflectivityModels(q=q_ref, name=name,
                                              z_left=config['z_left'],
                                              z_right=config['z_right'],
                                              dz=config['dz'])
    m.model_description =  model
    m.parameters = parameters

    # Option 1 is to create the data
    if create:
        print("Creating training data")
        t_0 = time.time()
        if use_errors:
            training_set(training_dir, m, n_train=n_train, errors=data_off[2]/data_off[1])
        else:
            training_set(training_dir, m, n_train=n_train)
        print("Training data created in %g sec" % (time.time()-t_0))
    # Option 2 is to train the network after loading the data
    else:
        print("Loading training data")
        q, train_data, train_pars = m.load(training_dir)

        print(train_data.shape)
        print(train_pars.shape)

        model = variational_model(train_data, train_pars,
                          z_left=config['z_left'], z_right=config['z_right'], dz=config['dz'])
        _n_train = min(n_train, train_data.shape[0]-v_size)
        history = model.fit(train_data[:_n_train], train_pars[:_n_train],
                            epochs=epoch, batch_size=4048,
                            validation_data=(train_data[-v_size:], train_pars[-v_size:]))

        save_model(model, history, name, training_dir)


def save_model(model, history, name, training_dir):
    # Save the trained model
    network.save_model(model, name, data_dir=training_dir)
    json.dump(history.history, open(os.path.join(training_dir, "%s_history.json" % name), 'w'))


def training_set(training_dir, m, n_train, errors=None):
    """
        Create a training set
        TODO add statistical fluctuation option
            errors = data_off[2]/data_off[1]
            test_pars, test_data = m.get_preprocessed_data(errors=errors)
    """
    # Create output directory as needed
    training_dir = os.path.abspath(training_dir)
    if not os.path.isdir(training_dir):
        print("Creating directory: %s" % training_dir)
        os.makedirs(training_dir)
    print("Generating...")
    m.generate(n_train)
    print("Pre-processing...")
    if errors is not None:
        train_pars, train_data = m.get_preprocessed_data(errors=errors)
    else:
        train_pars, train_data = m.get_preprocessed_data()
    print("Saving...")
    m.save(training_dir)
    return train_pars, train_data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Reflectometry ML")
    parser.add_argument('-n', metavar='training_size', type=int, default=1000, help="Training set size", dest='t_size', required=False)
    parser.add_argument('-v', metavar='validation_size', type=int, default=100, help="Validation set size", dest='v_size', required=False)
    parser.add_argument('-f', metavar='config_file', default="config.json", help="Configuration file", dest='config_file', required=False)
    parser.add_argument('--create', help="Create a new training set", action="store_true")
    parser.add_argument('-s', metavar='seed', type=int, default=42, help="Random seed", dest='r_seed', required=False)

    ns = parser.parse_args()

    # Load configuration
    with open(ns.config_file, 'r') as fd:
        config = json.load(fd)

    # Add source to path
    sys.path.append(config['src_dir'])
    sys.path.append(os.path.join(config['src_dir'], 'src'))

    import reflectivity_model
    import network
    from network import variational_model

    np.random.seed(ns.r_seed)

    # Make a copy of the config used
    json.dump(config, open(os.path.join(config['train_dir'], "%s_config.json" % config['name']), 'w'))

    workflow(config,
             n_train=ns.t_size,
             v_size=ns.v_size,
             use_errors=False,
             create=ns.create)
