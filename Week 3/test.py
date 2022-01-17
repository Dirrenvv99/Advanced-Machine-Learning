import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from makedata_python import make_data
import argparse
from tabulate import tabulate
import time
import json

np.random.seed(42)

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

w = np.loadtxt('w500')
beta_init_AK = 0.001
beta_init_exp = 0.1
N_runs = 20 #1 for test, 20 for 

parser = argparse.ArgumentParser(description= 'SA Opitimization of Ising model energy.')
parser.add_argument('-s',type= int, default= 0, help='0: AK schedule; 1 or higher: exponential schedule')
args = parser.parse_args()

exp_schedule = False
if args.s > 0:
    exp_schedule = True

def E(diff_array):
    return -0.25 * np.sum(diff_array)

# def E_dif(E_array, site):
#     res = 2 * np.sum(E_array[site])
#     return res

# def a_value(E_array,difbeta, site):
#     starting  = time.time()
#     diff = E_dif(E_array,site)
#     stopping = time.time()
#     run = stopping - starting
#     return np.exp(-1*beta *diff), diff, run

def a_iteration(site,diff_array, E_array, beta):
    diff = diff_array[site]
    real_diff = 0
    a = np.exp(-1*beta*diff)
    if a >= 1:            
        real_diff = diff
        E_array[site] = -1 * E_array[site]
        E_array[:,site] = E_array[:,site] * -1
        diff_array += 4 * E_array[:,site]
        diff_array[site] = -1 * diff_array[site]
    else:
        if np.random.random() < a:    
            real_diff = diff            
            E_array[site] = -1 * E_array[site]
            E_array[:,site] = E_array[:,site] * -1
            diff_array += 4 * E_array[:,site]
            diff_array[site] = -1 * diff_array[site]
    return real_diff

def MH_try(E_array, diff_array, beta,sites, energy):
    energies = []   
    start_a = time.time()

    diffs = np.array([a_iteration(site,diff_array,E_array,beta) for site in sites])
    energies = np.cumsum(diffs) + np.repeat(energy, len(sites))
#     for site in sites:
#         diff = diff_array[site]
#         a = np.exp(-1*beta*diff)
#         if a >= 1:            
#             energy = energy + diff
#             E_array[site] = -1 * E_array[site]
#             E_array[:,site] = E_array[:,site] * -1
#             diff_array += 4 * E_array[:,site]
#             diff_array[site] = -1 * diff_array[site]
#         else:
#             if np.random.random() < a:                
#                 energy = energy + diff
#                 E_array[site] = -1 * E_array[site]
#                 E_array[:,site] = E_array[:,site] * -1
#                 diff_array += 4 * E_array[:,site]
#                 diff_array[site] = -1 * diff_array[site]
#         energies.append(energy)
    stop_a = time.time()
    a_run = stop_a - start_a
    return E_array, energies, diff_array, a_run

def MH(E_array, diff_array, beta,sites, energy):
    energies = []   
    start_a = time.time()
    for site in sites:
        diff = diff_array[site]
        a = np.exp(-1*beta*diff)
        if a >= 1:            
            energy = energy + diff
            E_array[site] = -1 * E_array[site]
            E_array[:,site] = E_array[:,site] * -1
            diff_array += 4 * E_array[:,site]
            diff_array[site] = -1 * diff_array[site]
        else:
            if np.random.random() < a:                
                energy = energy + diff
                E_array[site] = -1 * E_array[site]
                E_array[:,site] = E_array[:,site] * -1
                diff_array += 4 * E_array[:,site]
                diff_array[site] = -1 * diff_array[site]
        energies.append(energy)
    return E_array, energies, diff_array
    
# def determine_initial_beta(diff_array):
#     #Notice that this implementation just tries all different spin flip possibilities, which is practically the same as proposed within the slides
#     highest_diff = -1 * np.inf
#     for site in range(0,w.shape[0]):
#         diff = diff_array[site]
#         if diff > highest_diff:
#             highest_diff = diff
#     return 1/highest_diff

def create_E_and_diff_array(x):
    E_array = np.empty_like(w)
    for index, _ in enumerate(E_array):
        E_array[index] = np.multiply(w[index],x) * x[index]
    diff_array = np.array([2 * np.sum(E_array[i]) for i,_ in enumerate(E_array)])
    return E_array, diff_array

def SK(L, AK=True, del_beta=0, f=0):
    var_E = np.inf
    means = []
    x = np.random.randint(0,2,size = w.shape[0])
    x[x == 0] = -1 
    E_array, diff_array = create_E_and_diff_array(x)
    energy = E(diff_array)
    beta = 1/np.max(diff_array)
    counter = 0
    vars = []
    betas= []
    while var_E > 0:
        if AK:
            beta = beta  + del_beta/np.sqrt(var_E)
        else:
            beta = beta * f
        betas.append(beta)
        counter += 1
        sites = np.random.randint(0,w.shape[0], size = L)
        E_array, energies, diff_array = MH(E_array,diff_array,beta,sites, energy)       
        energy = energies[-1]
        means.append(np.mean(energies))
        var_E = np.std(energies)**2
        vars.append(var_E)
    if AK:
        with open("energy_SA_AK_DB{}.json".format(del_beta), 'w') as f:
            json.dump({
                'del_beta:': del_beta,
                'L': L,
                'means': means,
                'betas:': betas,
                'stds: ': np.sqrt(vars)
                }, f, cls = NumpyEncoder)
    else:
        with open("energy_SA_EXP_f{}.json".format(f), 'w') as f:
            json.dump({
                'f:': f,
                'L': L,
                'means': means,
                'betas:': betas,
                'stds: ': np.sqrt(vars)
                }, f, cls = NumpyEncoder)

    
    # print("\n", counter)
    # plt.plot([x for x in range(counter)],betas)
    # plt.show()
    return means, betas, vars

def main():
    # del_betas = [0.1, 0.01, 0.001]
    del_betas = [0.1, 0.01, 0.001, 0.001]
    fs = [1.01, 1.001, 1.0002, 1.0002]
    table = []
    L = 500
    if exp_schedule:
        enum = fs
    else: 
        enum = del_betas
    fig, ax = plt.subplots(len(enum),3, constrained_layout = True)
    for index, value in enumerate(enum):
        if index == (len(enum) - 1):
            L = 1000
        min_values = []
        running_times = []
        for _ in tqdm(range(N_runs)):
            start_time = time.time()
            if exp_schedule:
                means, betas, vars = SK(L, AK=False, f = value)
            else:
                means, betas, vars = SK(L, AK=True, del_beta = value)
            stop_time = time.time()
            running_time = round(stop_time - start_time, 1)
            min_values.append(means[-1])
            running_times.append(running_time)
        min_energy = np.min(min_values)
        table.append((value, L, np.min(running_times), str(round(np.mean(min_values))) + " +- " + str(round(np.std(min_values), 0))))      
        print("minimal cost: \n", np.mean(min_values), " +- ", np.std(min_values))
        print("lowest reached value: \n", min_energy)
        # ax[index][0].plot(np.arange(0,len(means),1), means) #means plot
        # ax[index][1].plot(np.arange(0,len(vars), 1), stds) # std plot
        # ax[index][2].plot(np.arange(0,len(betas), 1), betas) #Betas plot
        # print(len(betas))
    # plt.show()
    if exp_schedule:
        with open("Full_Table_SA_exp.json", 'w') as f:
            json.dump({
                'full_table': table,
                'lowest_E' : min_energy
                }, f, cls = NumpyEncoder)    
    print(tabulate(table, tablefmt= "latex"))

        
# else: 
#     fs = [1.01, 1.001, 1.0002]
#     table = []
#     for index, f in enumerate(fs):
#         min_values = []
#         running_times = []
#         for _ in tqdm(range(N_runs)):
#             start_time = time.time()
#             means = SK_Exp(f, beta_init_exp, 500)
#             stop_time = time.time()
#             running_time = round(stop_time - start_time,1)
#             running_times.append(running_time)
#             min_values.append(means[-1])
#         table.append((f, 500, np.min(running_times), str(np.mean(min_values)) + " +- " + str(round(np.std(min_values), 0))))  
#         print("minimal cost: \n", np.mean(min_values), " +- ", np.std(min_values)) 
#         print("lowest reached value: \n", np.min(min_values))  
#         if index == len(fs) - 1:
#             min_values = []
#             running_times = []
#             for _ in tqdm(range(N_runs)):
#                 start_time = time.time()
#                 means = SK_Exp(f, beta_init_exp, 1000)
#                 stop_time = time.time()
#                 running_time = round(stop_time - start_time,1)
#                 running_times.append(running_time)
#                 min_values.append(means[-1])
#             table.append((f, 1000, np.min(running_times), str(np.mean(min_values)) + " +- " + str(round(np.std(min_values), 0))))  
#             print("minimal cost: \n", np.mean(min_values), " +- ", np.std(min_values)) 
#             print("lowest reached value: \n", np.min(min_values))     
#     print(tabulate(table, tablefmt= "latex"))       


if __name__ == '__main__':
    main()





