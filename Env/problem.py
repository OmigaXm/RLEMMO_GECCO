from scipy.spatial import distance
import numpy as np
import torch
import os, sys
import time
import random
import math
import copy
from tqdm import *
from options import MyOptions
from Env.env import Env

class Problem:
    def __init__(self, funcs, opts):
        super(Problem, self).__init__()
        self.funcs_list = funcs
        self.env_len = len(self.funcs_list)
        self.opts = opts

        self.env_list = []
        self.is_end = np.zeros(self.env_len, dtype='bool')

    def reset(self):
        self.env_list = []
        self.is_end = np.zeros(self.env_len, dtype='bool')
        pstates = []
        for i in range(self.env_len):
            self.env_list.append(Env(self.funcs_list[i], self.opts))
            states = self.env_list[i].reset()
            pstates.append(states.copy())

        return pstates

    def step(self, pactions):
        newpnext_states = []
        newprewards = []
        for i in range(self.env_len):
            next_states, rewards, is_done = self.env_list[i].step(pactions[i])
            newpnext_states.append(next_states.copy())
            newprewards.append(rewards)
            self.is_end[i] = is_done

        return newpnext_states, newprewards, self.is_end


