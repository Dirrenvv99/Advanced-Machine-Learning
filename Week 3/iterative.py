import numpy as np
from numpy.lib.index_tricks import diag_indices_from
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
from makedata_python import make_data

#the make_data function generates the weight matrix in the same manner as the matlab code did
#Note the parameter ferro, which determines whether the weight matrix is (anti-)ferromagnetic.
#To run the code as intended, we need n = 500, but the n can be chosen.

rng = np.random.default_rng()
w = np.loadtxt("w500")

def E(x):
    return -0.5 * np.dot(np.dot(x,w),x)

def E_dif(x, site):
    return 2 * x[site] * np.dot(x,w)[site]


def R(x):
    '''returns array of possible spin flips'''
    result = np.full((len(x), len(x)), x)
    new_diag = -1 * result.diagonal()
    np.fill_diagonal(result, new_diag)
    return result



def iterative_method(K, L):
    results_full = []
    sites = np.random.randint(0,w.shape[0], size = L)
    for _ in range(K):
        x = np.random.randint(0,2,size = w.shape[0])
        x[x == 0] = -1        
        Energy = E(x)
        for site in sites:
            diff = E_dif(x,site)
            if diff < 0:
                Energy = Energy + diff
                x[site] = x[site] * -1
            
            # #neigh = R(x)
            # #x_new = neigh[np.random.randint(0,len(neigh))]
            # if E_old <= E_new:
            #     x[site] = x[site] * -1
            # else:
            #     E_old = E_new
        results_full.append([x, Energy])
    results_full = np.array(results_full)
    energies = results_full[:,1]
    results = results_full[:,0]
    min_sol = results[np.argmin(energies)]
    min_energy = np.min(energies)
    return  results, energies, min_sol, min_energy

def main():
    print("Ferromagnetic simulation:")
    #w = make_data(500, ferro = False)
    ks = np.linspace(20, 40, 3)
    L = 1000
    N_runs = 20
    for k in ks:
        results_end = []
        energys = []
        print("k: ", k)
        for _ in tqdm(range(N_runs)):
            results, energies, min_x, min_e = iterative_method(int(k),L)
            energys.append(min_e)
            results_end.append(min_x)
        print("Mean energy over ", N_runs, " : ", np.mean(energys), " +- ", np.std(energys))




if __name__ == '__main__':
    main()



