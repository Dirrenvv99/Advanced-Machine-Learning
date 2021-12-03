import numpy as np
from numpy.lib.index_tricks import diag_indices_from
from tqdm import tqdm
from makedata_python import make_data

#the make_data function generates the weight matrix in the same manner as the matlab code did
#Note the parameter ferro, which determines whether the weight matrix is (anti-)ferromagnetic.
#To run the code as intended, we need n = 500, but the n can be chosen.

rng = np.random.default_rng()
w = np.loadtxt("w500")
#w = make_data(500, ferro = False) anti-ferromagnetic n = 500 data
#w = make_data(500, ferro = True) ferromagnetic n = 500 data


def E(x):
    return -0.5 * np.dot(np.dot(x,w),x)

def E_dif(x, site):
    return 2 * x[site] * np.dot(w[site],x)

def iterative_finder(sites): 
    x = np.random.randint(0,2,size = w.shape[0])
    x[x == 0] = -1        
    Energy = E(x)
    for site in sites:
        diff = E_dif(x,site)
        if diff < 0:
            Energy = Energy + diff
            x[site] = x[site] * -1
    return (x, Energy)



def iterative_method(K, L):
    results_full = []
    sites = np.random.randint(0,w.shape[0], size = L)
    results_full = [iterative_finder(sites) for _ in range(K)]
    results_full = np.array(results_full)
    energies = results_full[:,1]
    results = results_full[:,0]
    min_sol = results[np.argmin(energies)]
    min_energy = np.min(energies)
    return  results, energies, min_sol, min_energy

def main():
    print("w500 simulation:") #denote title w.r.t. the weight matrix used.
    ks = np.array([20,100,200,500,1000,2000,4000]) #Change this to other values for a.) to see when K becomes big enough for reliable results.
    #watch that the K = 4000 might take more than 60 minutes of running (at least on Dirren's cpu)
    L = 5000 #'normal' value that closely matched results found within excercise.
    N_runs = 20
    for k in ks:
        energys = []
        print("k: ", k)
        for _ in tqdm(range(N_runs)):
            results, energies, min_x, min_e = iterative_method(int(k),L)
            energys.append(min_e)
        print("Mean energy over ", N_runs, " runs : ", np.mean(energys), " +- ", np.std(energys))
        print("lowest energy found: ", np.min(energys))




if __name__ == '__main__':
    main()



