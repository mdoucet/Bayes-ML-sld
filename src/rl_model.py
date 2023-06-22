"""
  TODO:
  - Read DREAM fit results into model object
  -
"""
import numpy as np
import gymnasium as gym

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation

import json
import refl1d
from refl1d.names import *

import fitting.model_utils


class SLDEnv(gym.Env):

    def __init__(self, initial_state_file, final_state_file=None, data=None, reverse=True):
        """
            Initial and final states are in chronological order. The reverse parameter
            will take care of swapping the start and end states.
        """
        super().__init__()
        if reverse:
            self.expt_file = final_state_file
            self.end_expt_file = initial_state_file
        else:
            self.expt_file = initial_state_file
            self.end_expt_file = final_state_file
        self.data = data
        self.reverse = reverse

        if data is None:
            self.q = np.logspace(np.log10(0.009), np.log10(0.2), num=150)
        else:
            self.q = data[0][0]

        # Set up the model
        self.setup_model()

        # The state will correspond to the [time interval i] / [number of time intervals]
        self.time_stamp = self.data.shape[0]-1 if self.reverse else 0
        self.time_increment = -1 if self.reverse else 1
        self.start_state = True

        # Determine action space, normalized between 0 and 1
        action_size = len(self.low_array)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=[action_size], dtype=np.float32)
        # Observation space is the timestamp
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32)

    def setup_model(self):
        self.ref_model = fitting.model_utils.expt_from_json_file(self.expt_file, self.q, set_ranges=True)
        if self.end_expt_file:
            self.end_model = fitting.model_utils.expt_from_json_file(self.end_expt_file, self.q, set_ranges=True)
        else:
            self.end_model = None
        _, self.refl = self.ref_model.reflectivity()
        self.get_model_parameters()

    def get_model_parameters(self):
        self.par_labels = []
        self.parameters = []
        self.end_parameters = []
        self.low_array = []
        self.high_array = []
        for i, layer in enumerate(self.ref_model.sample):
            if not layer.thickness.fixed:
                self.par_labels.append(str(layer.thickness))
                self.parameters.append(layer.thickness.value)
                self.low_array.append(layer.thickness.bounds.limits[0])
                self.high_array.append(layer.thickness.bounds.limits[1])
                if self.end_model:
                    self.end_parameters.append(self.end_model.sample[i].thickness.value)
            if not layer.interface.fixed:
                self.par_labels.append(str(layer.interface))
                self.parameters.append(layer.interface.value)
                self.low_array.append(layer.interface.bounds.limits[0])
                self.high_array.append(layer.interface.bounds.limits[1])
                if self.end_model:
                    self.end_parameters.append(self.end_model.sample[i].interface.value)
            if not layer.material.rho.fixed:
                self.par_labels.append(str(layer.material.rho))
                self.parameters.append(layer.material.rho.value)
                self.low_array.append(layer.material.rho.bounds.limits[0])
                self.high_array.append(layer.material.rho.bounds.limits[1])
                if self.end_model:
                    self.end_parameters.append(self.end_model.sample[i].material.rho.value)
            if not layer.material.irho.fixed:
                self.par_labels.append(str(layer.material.irho))
                self.parameters.append(layer.material.irho.value)
                self.low_array.append(layer.material.irho.bounds.limits[0])
                self.high_array.append(layer.material.irho.bounds.limits[1])
                if self.end_model:
                    self.end_parameters.append(self.end_model.sample[i].material.irho.value)
        self.parameters = np.asarray(self.parameters)
        self.end_parameters = np.asarray(self.end_parameters)
        self.low_array = np.asarray(self.low_array)
        self.high_array = np.asarray(self.high_array)
        self.normalized_parameters = 2 * ( self.parameters - self.low_array ) / (self.high_array - self.low_array) - 1
        self.normalized_end_parameters = 2 * ( self.end_parameters - self.low_array ) / (self.high_array - self.low_array) - 1

    def convert_action_to_parameters(self, parameters):
        """
            Convert parameters from action space to physics spaces
        """
        deltas = self.high_array - self.low_array
        return self.low_array + deltas * (parameters + 1.0) / 2.0

    def set_model_parameters(self, values):
        """
            Parameters are normalized from 0 to 1
        """
        counter = 0

        for i, layer in enumerate(self.ref_model.sample):
            if not layer.thickness.fixed:
                layer.thickness.value = values[counter]
                counter += 1
            if not layer.interface.fixed:
                layer.interface.value = values[counter]
                counter += 1
            if not layer.material.rho.fixed:
                layer.material.rho.value = values[counter]
                counter += 1
            if not layer.material.irho.fixed:
                layer.material.irho.value = values[counter]
                counter += 1
        if not len(values) == counter:
            print("Action length doesn't match model: %s %s" % (len(values), counter))
        self.ref_model.update()
        _, self.refl = self.ref_model.reflectivity()

    def step(self, action):
        truncated = False
        info = {}
        pars = self.convert_action_to_parameters(action)
        self.set_model_parameters(pars)

        # Compute reward
        idx = self.data[self.time_stamp, 2] > 0

        if np.any(np.isnan(self.refl[idx])):
            self.refl = np.ones(len(self.refl))
            #print("NaNs in reflectivity")
        reward = -np.sum( (self.refl[idx] - self.data[self.time_stamp][1][idx])**2 / self.data[self.time_stamp][2][idx]**2 ) / len(self.data[self.time_stamp][2][idx])

        if self.reverse:
            terminated = self.time_stamp <= 0
        else:
            terminated = self.time_stamp >= self.data.shape[0]-1

        # Move to the next time time_stamp
        self.time_stamp += self.time_increment
        state = self.time_stamp / (self.data.shape[0]-1)
        state = np.array([state], dtype=np.float32)

        # Add a term for the boundary conditions (first and last times)
        if self.start_state:
            reward -= self.data.shape[0] * np.sum( (action - self.normalized_parameters)**2 ) / len(self.normalized_parameters)
        if terminated and self.end_model:
            reward -= self.data.shape[0] * np.sum( (action - self.normalized_end_parameters)**2 ) / len(self.normalized_end_parameters)

        self.start_state = False

        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.setup_model()
        self.time_stamp = self.data.shape[0]-1 if self.reverse else 0
        state = self.time_stamp / (self.data.shape[0]-1)
        state = np.array([state], dtype=np.float32)
        self.start_state = True
        info = {}
        return state, info

    def render(self, action=0, reward=0):
        print(action)

    def plot(self, scale=1, newfig=True, errors=False):
        if newfig:
            fig = plt.figure(dpi=100)
        plt.plot(self.q, self.refl*scale)

        idx = self.data[self.time_stamp, 1] > 0
        if errors:
            plt.errorbar(self.q[idx], self.data[self.time_stamp][1][idx]*scale,
                         yerr=self.data[self.time_stamp][2][idx]*scale, label=str(self.time_stamp))
        else:
            plt.plot(self.q[idx], self.data[self.time_stamp][1][idx]*scale,
                     label=str(self.time_stamp))

        plt.gca().legend()
        plt.xlabel('Q [$1/\AA$]')
        plt.ylabel('R')
        plt.xscale('log')
        plt.yscale('log')
        plt.show()


from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        print("STEP!")
        return True
