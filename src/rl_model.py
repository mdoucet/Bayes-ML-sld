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

import reflectivity_model as rm


class SLDEnv(gym.Env):

    def __init__(self, ref_model):
        super().__init__()

        self.ref_model = ref_model

        # Determine action space
        low_array = []
        high_array = []
        for i, par in enumerate(self.ref_model.parameters):
            low_array.append(par['bounds'][0])
            high_array.append(par['bounds'][1])

            #self.model_description['layers'][par['i']][par['par']] = pars[i]

        self.action_space = gym.spaces.Box(low=np.array(low_array), high=np.array(high_array), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1.0, high=7.0, shape=(10, 2), dtype=np.float32)

        self.refl = []

    def step(self, action):
        state = 1
        reward = -1
        terminated = True
        truncated = False
        info = {}
        self.ref_model.compute_reflectivity([action])
        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = np.ones([10,2], dtype=np.float32)
        info = {}
        return state, info

    def render(self, action=0, reward=0):
        print(action)

    def plot(self):
        fig = plt.figure(dpi=100)
        plt.plot(self.ref_model.q, self.ref_model._refl_array[-1], label="Truth")

        plt.gca().legend()
        plt.xlabel('Q [$1/\AA$]')
        plt.ylabel('R')
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
