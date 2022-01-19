import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from makedata_python import make_data
import argparse
from tabulate import tabulate
import time
import json

#Encoding to be able to output numpy arrays directly to a JSON file
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
N_runs = 20 

parser = argparse.ArgumentParser(description= 'SA Opitimization of Ising model energy.')
parser.add_argument('-s',type= int, default= 0, help='0: AK schedule; 1 or higher: exponential schedule')
args = parser.parse_args()

exp_schedule = False
if args.s > 0:
    exp_schedule = True

def E(diff_array):
    return -0.25 * np.sum(diff_array)

def MH(E_array, diff_array, beta,sites, energy):
    energies = []   
    for site in sites:
        diff = diff_array[site]
        a = np.exp(-1*beta*diff)
        if a >= 1:            
            energy = energy + diff
            #Below are the updating steps for the energy and difference array
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

def create_E_and_diff_array(x):
    ''' Creates the energy and difference array'''
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
        with open("energy_SA_EXP_f{}.json".format(f), 'w') as file:
            json.dump({
                'f:': f,
                'L': L,
                'means': means,
                'betas:': betas,
                'stds: ': np.sqrt(vars)
                }, file, cls = NumpyEncoder)
    return means, betas, vars

def main():
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
    if exp_schedule:
        with open("Full_Table_SA_exp.json", 'w') as f:
            json.dump({
                'full_table': table,
                'lowest_E' : min_energy
                }, f, cls = NumpyEncoder)    
    else:
        with open("Full_Table_SA_AK.json", 'w') as f:
            json.dump({
                'full_table': table,
                'lowest_E' : min_energy
                }, f, cls = NumpyEncoder)  
    print(tabulate(table, tablefmt= "latex"))

if __name__ == '__main__':
    main()





