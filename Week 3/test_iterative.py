import numpy as np
from numpy.lib.index_tricks import diag_indices_from
from tabulate import tabulate
from tqdm import tqdm
from makedata_python import make_data
import argparse
import json
import time

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

#the make_data function generates the weight matrix in the same manner as the matlab code did
#Note the parameter ferro, which determines whether the weight matrix is (anti-)ferromagnetic.
#To run the code as intended, we need n = 500, but the n can be chosen.

parser = argparse.ArgumentParser(description= 'SA Opitimization of Ising model energy.')
parser.add_argument('--ferro',type= int, default= 2, help='0: Frustrated Problem; 1 : Ferromagnetic problem; 2 or higher: w500 problem')
args = parser.parse_args()


if args.ferro == 0:
    w = make_data(500, ferro = False)
    print("Frustrated problem instance")
elif args.ferro == 1:
    w = make_data(500, ferro = True)
    print("Ferromagnetic problem instance")
elif args.ferro ==2:
    w = np.loadtxt("w500")
    print("w500 problem")

rng = np.random.default_rng()

def E(diff_array):
    return -0.25 * np.sum(diff_array)

# def E_dif(x, site):
#     return 2 * x[site] * np.dot(w[site],x)

def create_E_and_diff_array(x):
    E_array = np.empty_like(w)
    for index, _ in enumerate(E_array):
        E_array[index] = np.multiply(w[index],x) *x[index]
    diff_array = np.array([2 * np.sum(E_array[i]) for i,_ in enumerate(E_array)])
    if args.ferro > 2:
        print(x)
        print(E_array)
    return E_array, diff_array

def iteration(E_array, diff_array, site):
    diff = diff_array[site]
    real_diff = 0
    if diff < 0:
        real_diff = diff
        E_array[site] = -1 * E_array[site]
        E_array[:,site] = E_array[:,site] * -1
        diff_array += 4 * E_array[:,site]
        diff_array[site] = -1 * diff_array[site]
    return real_diff
     
def iterative_finder(E_array, diff_array, L, K, json_print = False): 
    sites = np.random.randint(0,w.shape[0], size = L)
    # x = np.random.randint(0,2,size = w.shape[0])
    # x[x == 0] = -1      
    # E_array, diff_array = create_E_and_diff_array(x)  
    energy = E(diff_array)
    # differences = [iteration(E_array, diff_array, site) for site in sites]
    if json_print:
        energies = [energy]
    for site in sites:
        energy += iteration(E_array, diff_array, site)
        if json_print:
            energies.append(energy)
    if json_print:
        with open("energy_iterative_F{}_K{}.json".format(args.ferro,K), 'w') as f:
            json.dump({
                'K': K,
                'L': L,
                'energy': energies
                }, f, cls = NumpyEncoder)
    return energy



def iterative_method(E_array, diff_array , K, L, json_print = False):
    results_full = []
    roll_number = int(np.random.random() * w.shape[0])
    if json_print:
        results_full = [iterative_finder(np.roll(np.roll(E_array, i * roll_number, axis = 0), i * roll_number, axis = 1), np.roll(diff_array, i * roll_number), L, K) for i in range(K-1)]
        results_full.append(iterative_finder(np.roll(np.roll(E_array, (K-1) * roll_number, axis = 0), (K-1) * roll_number, axis = 1), np.roll(diff_array, (K-1) * roll_number), L, K, True))
    else:
        results_full = [iterative_finder(np.roll(np.roll(E_array, i * roll_number, axis = 0), i * roll_number, axis = 1), np.roll(diff_array, i * roll_number), L, K) for i in range(K)]    
    energies = np.array(results_full)
    min_energy = np.min(energies)
    return  energies, min_energy

def main():
    # ks = np.array([20])
    ks = np.array([20,100,200,500,1000,2000,4000])
    #watch that the K = 4000 on dirren CPU takes 100 * 20 seconds approx. For Olivier should be about 60 * 20 seconds (given previous results). For frustrated problems.
    if args.ferro == 1:
        L = 5000 #Lower value for ferromagnetic problem, chosen via plots, since this context will result in faster convergence
    else:
        L = 17500 #Lowest value with similar results for W500 with respect to the assignment
    N_runs = 20
    table = []
    x = np.random.randint(0,2,size = w.shape[0])
    x[x == 0] = -1  
    E_begin, D_begin = create_E_and_diff_array(x)
    for k in ks:
        E_array = np.roll(E_begin, int(k - k/100), axis = 0)
        E_array = np.roll(E_array, int(k - k/100), axis = 1)
        diff_array = np.roll(D_begin, int(k - k/100))
        energys = []
        run_times = []
        print("k: ", k)
        for r in tqdm(range(N_runs)):
            start_time = time.time()
            if r == N_runs - 1:
                energies, min_e = iterative_method(E_array, diff_array,int(k),L, True)
            else:
                energies, min_e = iterative_method(E_array, diff_array, int(k),L)
            stop_time = time.time()
            run_times.append(round(stop_time - start_time, 1))
            energys.append(min_e)
        min_energy  = np.min(energys)
        print("Mean energy over ", N_runs, " runs : ", np.mean(energys), " +- ", np.std(energys))
        print("lowest energy found: ", min_energy)
        table.append((k,np.min(run_times), str(round(np.mean(energys))) + " +- " + str(round(np.std(energys)))))
    with open("Full_Table_iterative_F{}.json".format(args.ferro), 'w') as f:
        json.dump({
            'full_table': table,
            'lowest_E' : min_energy
            }, f, cls = NumpyEncoder)    
    print(tabulate(table, tablefmt="latex"))


if __name__ == '__main__':
    main()



