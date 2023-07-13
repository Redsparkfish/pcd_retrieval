import time
import numpy as np
from scipy.optimize import linear_sum_assignment

dists = np.load('distances.npy')
rol_ind, col_ind = linear_sum_assignment(dists)
print(rol_ind.shape, col_ind.shape)
print(dists[rol_ind, col_ind].sum()/rol_ind.shape[0])

dists1 = np.load('distances1.npy')
rol_ind1, col_ind1 = linear_sum_assignment(dists1)
print(dists[rol_ind1, col_ind1].sum()/rol_ind.shape[0])
