
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
import numpy as np
import os, sys

def mydbscan(env):
    
    pop = env.cur_GA.population.copy()
    pop = (pop - env.lowbound) / (env.upperbound -env.lowbound)

    # eps = pow(env.min_samples / ((2*env.dim+1) / pow(1 / pow(env.popsize, 1/env.dim), env.dim)), 1/env.dim)
    eps = 0.2

    clustering = DBSCAN(eps = eps, min_samples = env.min_samples).fit(pop)
    cluster_labels = clustering.labels_

    return cluster_labels


