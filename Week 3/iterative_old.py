import numpy as np
from numpy.lib.index_tricks import diag_indices_from
from tabulate import tabulate
from tqdm import tqdm
from makedata_python import make_data
import argparse
import time

#the make_data function generates the weight matrix in the same manner as the matlab code did
#Note the parameter ferro, which determines whether the weight matrix is (anti-)ferromagnetic.
#To run the code as intended, we need n = 500, but the n can be chosen.

parser = argparse.ArgumentParser(description= 'SA Opitimization of Ising model energy.')
parser.add_argument('--ferro',type= int, default= 0, help='0: Frustrated Problem; 1 : Ferromagnetic problem; 2 or higher: w500 problem')
args = parser.parse_args()


if args.ferro == 0:
    w = make_data(500, ferro = False)
    print("Frustrated problem instance")
elif args.ferro == 1:
    w = make_data(500, ferro = True)
    print("Ferromagnetic problem instance")
else:
    w = np.loadtxt("w500")
    print("w500 problem")
rng = np.random.default_rng()


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
    ks = np.array([20,100])
    # ks = np.array([20,100,200,500,1000,2000,4000]) #Change this to other values for a.) to see when K becomes big enough for reliable results.
    #watch that the K = 4000 might take more than 60 minutes of running (at least on Dirren's cpu)
    L = 5000 #'normal' value that closely matched results found within excercise.
    N_runs = 20
    table = []
    for k in ks:
        energys = []
        run_times = []
        print("k: ", k)
        for _ in tqdm(range(N_runs)):
            start_time = time.time()
            results, energies, min_x, min_e = iterative_method(int(k),L)
            stop_time = time.time()
            run_times.append(round(stop_time - start_time, 1))
            energys.append(min_e)
        print("Mean energy over ", N_runs, " runs : ", np.mean(energys), " +- ", np.std(energys))
        print("lowest energy found: ", np.min(energys))
        table.append((k,np.min(run_times), str(np.mean(energys)) + " +- " + str(np.std(energys))))
    print(tabulate(table, tablefmt="latex"))


if __name__ == '__main__':
    main()



