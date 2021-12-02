import numpy as np
from numpy.lib.index_tricks import diag_indices_from
import scipy.stats as stats
import scipy.sparse as sparse
import matplotlib.pyplot as plt

np.random.seed(20)

def sprandsym(n, density):
    rvs = stats.norm().rvs
    X = sparse.random(n, n, density=density, data_rvs=rvs)
    upper_X = sparse.triu(X) 
    result = upper_X + upper_X.T - sparse.diags(X.diagonal())
    return result.toarray()

def make_data(n, ferro = True):
    w = sprandsym(n,1)
    if ferro == False:
        w[w>0] = 1
        w[w<0] = -1
    else:
        w[w>0] = 1
        w[w<0] = 0
    w[diag_indices_from(w)] = 0
    return w 






