import numpy as np
from scipy.spatial import distance

class GENETICALGORITHM:
    def __init__(self, no_dimension, lbound, ubound, FF, CR, popsize, gbest):
        super(GENETICALGORITHM, self).__init__()
        self.dim = no_dimension
        self.low_bound = lbound
        self.upper_bound = ubound
        self.FF = FF
        self.CR = CR
        self.population_size = popsize
        self.max_dist = distance.euclidean(self.low_bound, self.upper_bound)
        self.gbest = gbest

        self.population = None
        self.val = None
        self.parent_val = None
        self.best_so_far = None
        self.worst_so_far = None
        self.vornoi_matrix = None
        self.best_so_far_pos = None
        self.local_best = [None] * self.population_size
        self.local_best_pos = [None] * self.population_size
        

def initial(GA, func):
    
    GA.population = np.random.rand(GA.population_size, GA.dim) * np.repeat(np.expand_dims((GA.upper_bound - GA.low_bound), 0), GA.population_size, 0) + np.repeat(np.expand_dims((GA.low_bound), 0), GA.population_size, 0) # population_pos
    replace1 = np.where(GA.population < np.repeat(np.expand_dims(GA.low_bound, 0), GA.population_size, 0))
    GA.population[replace1] = np.repeat(np.expand_dims(GA.low_bound, 0), GA.population_size, 0)[replace1]
    replace2 = np.where(GA.population > np.repeat(np.expand_dims(GA.upper_bound, 0), GA.population_size, 0))
    GA.population[replace2] = np.repeat(np.expand_dims(GA.upper_bound, 0), GA.population_size, 0)[replace2]
    GA.val = -func.evaluate(GA.population.copy()) + GA.gbest # maximum problem
    GA.best_so_far = np.min(GA.val)
    GA.best_so_far_pos = GA.population[np.argmin(GA.val)].copy()
    GA.worst_so_far = np.max(GA.val)
    GA.parent_val = GA.val.copy()