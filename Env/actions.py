import time
import numpy as np
import torch
from scipy.spatial import distance

def find_nei(GA, device, k_neis): # todo: be torch
    population_dist = distance.cdist(GA.population, GA.population)
    population_dist[range(GA.population_size), range(GA.population_size)] = GA.max_dist
    population_dist_arg = np.argsort(population_dist.copy(), axis = -1)

    GA.vornoi_matrix = np.zeros(shape=(GA.population_size, GA.population_size))
    for i in range(GA.population_size):
        GA.vornoi_matrix[i][population_dist_arg[i, :k_neis]] = 1


def act1(GA, pop_choice):
    stacked_pos = GA.population[pop_choice].copy()
    sigmas = np.float_power(10,-np.random.randint(0, 9, GA.dim))
    v = stacked_pos + np.random.normal(loc=0, scale=sigmas)
    return v

def act2(GA, pop_choice):
    stacked_pos = GA.population[pop_choice].copy()
    stacked_rs = np.zeros((len(pop_choice), 2), dtype='int')
    stacked_best = np.zeros(len(pop_choice), dtype='int')
    for idx in range(len(pop_choice)):
        neibor = np.argwhere(GA.vornoi_matrix[pop_choice[idx]] > 0).squeeze(-1)
        stacked_rs[idx, :] = np.random.choice(neibor, 2, replace=False)
        niching = np.append(neibor.copy(),pop_choice[idx])
        stacked_best[idx] = niching[np.argmin(GA.val[niching])]

    r1 = stacked_rs[:, 0]
    r2 = stacked_rs[:, 1]
    v = stacked_pos + GA.FF * (GA.population[stacked_best] - stacked_pos) + GA.FF * (GA.population[r1] - GA.population[r2])
    return v

def act3(GA, pop_choice):
    stacked_pos = GA.population[pop_choice].copy()
    stacked_rs = np.zeros((len(pop_choice), 3), dtype='int')
    for idx in range(len(pop_choice)):
        neibor = np.argwhere(GA.vornoi_matrix[pop_choice[idx]] > 0).squeeze(-1)
        stacked_rs[idx, :] = np.random.choice(neibor, 3, replace=False)

    r1 = stacked_rs[:, 0]
    r2 = stacked_rs[:, 1]
    r3 = stacked_rs[:, 2]
    v = GA.population[r1] + GA.FF * (GA.population[r2] - GA.population[r3])
    return v

def act4(GA, pop_choice):
    stacked_pos = GA.population[pop_choice].copy()
    stacked_rs = np.zeros((len(pop_choice), 2), dtype='int')
    stacked_best = np.zeros(len(pop_choice), dtype='int')
    for idx in range(len(pop_choice)):
        stacked_rs[idx, :] = np.random.choice(np.delete(np.arange(GA.population_size), pop_choice[idx]), 2, replace=False)

    for idx in range(len(pop_choice)):
        random_best = np.random.choice(np.delete(np.arange(GA.population_size), pop_choice[idx]), 1, replace=False)[0]
        neibor = np.argwhere(GA.vornoi_matrix[random_best] > 0).squeeze(-1)
        niching = np.append(neibor.copy(),random_best)
        stacked_best[idx] = niching[np.argmin(GA.val[niching])]

    r1 = stacked_rs[:, 0]
    r2 = stacked_rs[:, 1]
    v = stacked_pos + GA.FF * (GA.population[stacked_best] - stacked_pos) + GA.FF * (GA.population[r1] - GA.population[r2])
    return v

def act5(GA, pop_choice):
    stacked_pos = GA.population[pop_choice].copy()
    stacked_rs = np.zeros((len(pop_choice), 3), dtype='int')
    for idx in range(len(pop_choice)):
        stacked_rs[idx, :] = np.random.choice(np.delete(np.arange(GA.population_size), pop_choice[idx]), 3, replace=False)

    r1 = stacked_rs[:, 0]
    r2 = stacked_rs[:, 1]
    r3 = stacked_rs[:, 2]
    v = GA.population[r1] + GA.FF * (GA.population[r2] - GA.population[r3])
    return v
