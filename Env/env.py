from scipy.spatial import distance
import numpy as np
import torch
from tensorboard_logger import Logger as TbLogger
import os, sys
import time
import random
import math
import copy
from  Env.actions import *
from utils import clip_grad_norms
from Env.GA import *
from cec2013.cec2013 import *
from tqdm import *
from options import MyOptions
from mydbscan import *

def getState(env): # bests无独立第二维
    GA = env.cur_GA
    states = np.zeros((GA.population_size, env.node_dim))
    assert env.node_dim == 22
    all_dist = distance.cdist(GA.population, GA.population)
    states[:, 0] = np.average(np.sum(all_dist, -1) / (env.popsize - 1)) / GA.max_dist
    states[:, 1] = np.std((GA.val) / (GA.worst_so_far))
    states[:, 2] = ((env.max_FE / env.popsize) - (env.FE / env.popsize)) / (env.max_FE / env.popsize)
    states[:, 3] = (env.stagnation) / (env.max_FE / env.popsize)
    states[:, 4] = np.average(GA.val / GA.worst_so_far)

    sub_optimal = np.zeros(env.popsize)
    for idx in range(GA.population_size):
        neibor = np.argwhere(GA.vornoi_matrix[idx] > 0).squeeze(-1)
        niching = np.append(neibor.copy(),idx)
        assert len(neibor) == 4
        assert len(niching) == len(neibor) + 1
        best = niching[np.argmin(GA.val[niching])]
        niching_dist = distance.cdist(GA.population[niching], GA.population[niching])
        sub_optimal[idx] = np.min(GA.val[niching])
        if GA.local_best[idx] == None:
            GA.local_best[idx] = sub_optimal[idx]
            GA.local_best_pos[idx] = GA.population[best].copy()
            env.local_stagnation[idx] = 0
        else:
            if sub_optimal[idx] < GA.local_best[idx]:
                GA.local_best[idx] = sub_optimal[idx]
                GA.local_best_pos[idx] = GA.population[best].copy()
                env.local_stagnation[idx] = 0
            else:
                env.local_stagnation[idx] += 1


        states[idx, 5] = np.average(np.sum(niching_dist) / (len(niching) - 1)) / GA.max_dist
        states[idx, 6] = np.std((GA.val[niching]) / (GA.worst_so_far))
        states[idx, 7] = env.local_stagnation[idx] / (env.max_FE / env.popsize)
        states[idx, 8] = np.average(GA.val[niching] / GA.worst_so_far)
        
        states[idx, 14] = distance.euclidean(GA.population[idx], GA.population[best]) / GA.max_dist
        states[idx, 15] = (GA.val[idx] - np.min(GA.val[niching])) / (GA.worst_so_far) 

        states[idx, 18] = np.average((GA.val[idx] - GA.val[neibor]) / GA.worst_so_far)
        in_nich_dist =  distance.cdist([GA.population[idx]], GA.population[neibor])[0]
        states[idx, 19] = np.average(in_nich_dist) / GA.max_dist
        states[idx, 20] = np.sum(GA.val[idx] - GA.val) / (env.popsize -1) / GA.worst_so_far

    sub_rank = np.argsort(sub_optimal)
    for idx in range(GA.population_size):
        states[idx, 9] = np.where(sub_rank == idx)[0][0] / (env.popsize - 1)

    states[:, 10] = distance.cdist([GA.population[np.argmin(GA.val)]], GA.population)[0] / GA.max_dist
    states[:, 11] = distance.cdist([GA.best_so_far_pos], GA.population)[0] / GA.max_dist
    states[:, 12] = (GA.val - GA.best_so_far) / (GA.worst_so_far)
    states[:, 13] = (GA.val - np.min(GA.val)) / (GA.worst_so_far)

    states[:, 16] = env.limit / (env.max_FE / env.popsize)
    states[:, 17] = GA.val / GA.worst_so_far

    states[:, 21] = np.sum(all_dist, -1) / (env.popsize - 1) / GA.max_dist

    assert not (True in np.isnan(states))
    return states



class Env:
    def __init__(self, func, opts):
        super(Env, self).__init__()
        # constant
        self.node_dim = opts.node_dim
        self.FF = opts.FF
        self.CR = opts.CR
        self.popsize = opts.popsize
        self.func = func
        self.dim = self.func.get_dimension()
        self.lowbound = np.zeros(self.dim)
        self.upperbound = np.zeros(self.dim)
        for k in range(self.dim):
            self.upperbound[k] = self.func.get_ubound(k)
            self.lowbound[k] = self.func.get_lbound(k)
        self.max_FE = self.func.get_maxfes()
        self.n_action = opts.n_action
        self.mutations = [act1, act2, act3, act4, act5]
        self.gbest = self.func.get_fitness_goptima()
        self.device = opts.device
        self.min_samples = opts.min_samples
        self.k_neis = opts.k_neis
        

        # to maintain
        self.cur_GA =  None
        self.FE = 0
        self.archive_pos = None
        self.archive_val = None
        self.is_done = False

        # maintain for log
        self.rewardp1 = 0
        self.rewardp2 = 0
        self.rewardp3 = 0
        self.rewards = 0
        self.returnsp1 = 0
        self.returnsp2 = 0
        self.returnsp3 = 0
        self.returns = 0
        self.chosen_actions = np.zeros(self.n_action)
        self.reini_count = 0
        self.labels = None

        self.stagnation = 0
        self.local_stagnation = np.zeros(self.popsize)
        self.limit = np.zeros(self.popsize)


    def reset(self):
        self.archive_pos = []
        self.archive_val = []
        self.cur_GA = GENETICALGORITHM(self.dim, self.lowbound, self.upperbound, self.FF, self.CR, self.popsize, self.gbest)
        initial(self.cur_GA, self.func)
        self.FE += self.popsize

        for to_store in range(self.popsize):
            if self.cur_GA.val[to_store] - self.cur_GA.best_so_far <= 0.1:
                self.archive_pos.append(self.cur_GA.population[to_store].copy())
                self.archive_val.append(self.cur_GA.val[to_store])

        # self.labels = mydbscan(self)
        self.cur_GA.vornoi_matrix = np.zeros(shape=(self.popsize, self.popsize))
        find_nei(self.cur_GA, self.device, self.k_neis)
        assert not (self.cur_GA.vornoi_matrix == 0).all()
        

        states= getState(self) # popsize, neis, fea_dim

        return states

    
    
    def step(self, actions): # action.shape (popsize)

        bprimes = np.zeros((self.popsize, self.dim))
        actcount = 0
        for action in range(self.n_action):
            pop_choice = np.where(actions == action)[0]
            actcount += len(pop_choice)
            if len(pop_choice) == 0:
                continue
            self.chosen_actions[action] += len(pop_choice)
            mutate = self.mutations[action]
            bprime = mutate(self.cur_GA, pop_choice)
            bprimes[pop_choice] = bprime.copy()
        assert actcount == self.popsize

        NP, Ndim = self.popsize, self.dim
        jrand = np.random.randint( Ndim, size=(NP))
        tmp_pos= np.where(np.random.rand(NP, Ndim) < self.CR, bprimes, self.cur_GA.population)
        tmp_pos[np.arange(NP), jrand] = bprimes[np.arange(NP), jrand].copy()

        replace1 = np.where(tmp_pos < np.repeat(np.expand_dims(self.lowbound, 0), self.popsize, 0))
        tmp_pos[replace1] = np.repeat(np.expand_dims(self.lowbound, 0), self.popsize, 0)[replace1]
        replace2 = np.where(tmp_pos > np.repeat(np.expand_dims(self.upperbound, 0), self.popsize, 0))
        tmp_pos[replace2] = np.repeat(np.expand_dims(self.upperbound, 0), self.popsize, 0)[replace2]

        tmp_val = -self.func.evaluate(tmp_pos.copy()) + self.gbest # maximum problem
        self.FE += self.popsize
		
        isbigger = np.zeros(self.popsize, dtype='bool')
        for i in range(self.popsize):
            if tmp_val[i] <= self.cur_GA.val[i] and (not (tmp_pos[i] == self.cur_GA.population[i]).all()):
                isbigger[i] = True
        bigger = np.where(isbigger == True)[0]
        smaller = np.where(isbigger == False)[0]
        self.cur_GA.population[bigger] = np.copy(tmp_pos[bigger])
        self.cur_GA.val[bigger] = tmp_val[bigger].copy()

        self.limit[bigger] = 0
        self.limit[smaller] += 1
        if np.min(self.cur_GA.val) < self.cur_GA.best_so_far:
            self.cur_GA.best_so_far = np.min(self.cur_GA.val)
            self.cur_GA.best_so_far_pos = self.cur_GA.population[np.argmin(self.cur_GA.val)].copy()
            self.stagnation = 0
        else:
            self.stagnation += 1
        
        for to_store in bigger:
            if self.cur_GA.val[to_store] - self.cur_GA.best_so_far <= 0.1:
                self.archive_pos.append(self.cur_GA.population[to_store].copy())
                self.archive_val.append(self.cur_GA.val[to_store])
		
        self.reinitial()
        self.cur_GA.parent_val = self.cur_GA.val.copy()
		
        self.labels = mydbscan(self)

        self.rewards = 0
        self.rewardp1 = 0
        self.rewardp2 = 0
        self.rewardp3 = 0
        for label in range(np.max(self.labels) + 1):
            now_cluster = np.where(self.labels == label)[0]
            minval = np.min(self.cur_GA.val[now_cluster])
            self.rewardp1 += (1 - minval / self.cur_GA.worst_so_far)
        
        self.rewards = self.rewardp1 + self.rewardp2 + self.rewardp3
        self.rewards /= 1000
        self.returnsp1 += self.rewardp1
        self.returnsp2 += self.rewardp2
        self.returnsp3 += self.rewardp3        
        self.returns += self.rewards

        self.cur_GA.vornoi_matrix = np.zeros(shape=(self.popsize, self.popsize))
        find_nei(self.cur_GA, self.device, self.k_neis)
        assert not (self.cur_GA.vornoi_matrix == 0).all()
        #assert ((self.cur_GA.vornoi_matrix.T == self.cur_GA.vornoi_matrix).all())
        
        next_states = getState(self) 

        if self.FE >= self.max_FE:
            self.is_done = True

        return next_states, self.rewards, self.is_done

    def reinitial(self):
        to_reini = []
        for i in range(self.popsize):
            if ((self.cur_GA.val[i]- self.cur_GA.best_so_far < 1e-7 and self.limit[i] > 0.1*self.popsize) or self.limit[i] > self.popsize):
                to_reini.append(i)
                self.limit[i] = 0
                self.cur_GA.local_best[i] = None
                self.cur_GA.local_best_pos[i] = None
                self.local_stagnation[i] = 0

        if len(to_reini) > 0:
            reini_len = len(to_reini)
            self.reini_count += reini_len
            self.cur_GA.population[to_reini] = np.random.rand(reini_len, self.dim) * np.repeat((self.upperbound - self.lowbound)[None,:], reini_len, 0) + np.repeat(self.lowbound[None,:], reini_len, 0)
            replace1 = np.where(self.cur_GA.population < np.repeat(np.expand_dims(self.lowbound, 0), self.popsize, 0))
            self.cur_GA.population[replace1] = np.repeat(np.expand_dims(self.lowbound, 0), self.popsize, 0)[replace1]
            replace2 = np.where(self.cur_GA.population > np.repeat(np.expand_dims(self.upperbound, 0), self.popsize, 0))
            self.cur_GA.population[replace2] = np.repeat(np.expand_dims(self.upperbound, 0), self.popsize, 0)[replace2]
            
            self.cur_GA.val[to_reini] = -self.func.evaluate(self.cur_GA.population[to_reini].copy()) + self.gbest
            self.FE += reini_len

            if np.min(self.cur_GA.val) < self.cur_GA.best_so_far:
                self.cur_GA.best_so_far = np.min(self.cur_GA.val)
                self.cur_GA.best_so_far_pos = self.cur_GA.population[np.argmin(self.cur_GA.val)].copy()
                self.stagnation = 0

            for i in to_reini:
                if self.cur_GA.val[i] - self.cur_GA.best_so_far <= 0.1:
                    self.archive_pos.append(self.cur_GA.population[i].copy())
                    self.archive_val.append(self.cur_GA.val[i])

            if np.max(self.cur_GA.val) > self.cur_GA.worst_so_far:
                self.worst_so_far =  np.max(self.cur_GA.val)
