import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from makedata_python import make_data
import argparse

# w = make_data(5,False)
w = np.loadtxt("w500")
beta_init = 0.001
N_runs = 20

parser = argparse.ArgumentParser(description= 'SA Opitimization of Ising model energy.')
parser.add_argument('-s',type= int, default= 0, help='0: AK schedule; 1 or higher: exponential schedule')
args = parser.parse_args()

exp_schedule = False
if args.s > 0:
    exp_schedule = True

def E(x):
    return -0.5 * np.dot(np.dot(x,w),x)

def E_dif(x, site):
    return 2 * x[site] * np.dot(w[site],x)

def a_value(x,beta, site):
    diff = E_dif(x,site)
    return np.exp(-1*beta *diff), diff

def MH(x,beta,sites, energy):
    energies = []    
    for site in sites:
        a, diff = a_value(x,beta, site)
        # print("a: ", a)
        # print("diff: ", diff)
        if a >= 1:
            x[site] = -1 * x[site]
            energy = energy + diff
        else:
            if np.random.random() < a:
                x[site] = -1*x[site]
                energy = energy + diff
        # print(energy)
        energies.append(energy)
    return x, energies
    

def SK_AK(del_beta, beta_init, L):
    var_E = 1
    means = []
    eps = 0.0005
    beta = beta_init
    x = np.random.randint(0,2,size = w.shape[0])
    x[x == 0] = -1 
    energy = E(x)
    while var_E > eps:
        sites = np.random.randint(0,w.shape[0], size = L)
        beta = beta  + del_beta/np.sqrt(var_E)
        x, energies = MH(x,beta,sites, energy)
        energy = energies[-1]
        means.append(np.mean(energies))
        var_E = np.std(energies)
        # print(var_E)
    return means

def SK_Exp(f, beta_init, L):
    var_E = 1
    means = []
    eps = 0.0005
    beta = beta_init
    x = np.random.randint(0,2,size = w.shape[0])
    x[x == 0] = -1 
    energy = E(x)
    while var_E > eps:
        sites = np.random.randint(0,w.shape[0], size = L)
        beta = beta* f
        x, energies = MH(x,beta,sites, energy)
        energy = energies[-1]
        means.append(np.mean(energies))
        var_E = np.std(energies)
        # print(var_E)
    return means


def main():
    if exp_schedule == False:
        del_betas = [0.1, 0.01, 0.001]
        for del_beta in del_betas:
            min_values = []
            for _ in tqdm(range(N_runs)):
                means = SK_AK(del_beta, beta_init, 500)
                min_values.append(means[-1])
            print("minimal cost: \n", np.mean(min_values), " +- ", np.std(min_values))
            print("lowest reached value: \n", np.min(min_values))
    else: 
        fs = [1.01,1.001,1.0002]
        for f in fs:
            min_values = []
            for _ in tqdm(range(N_runs)):
                means = SK_Exp(f, beta_init, 500)
                min_values.append(means[-1])
            print("minimal cost: \n", np.mean(min_values), " +- ", np.std(min_values)) 
            print("lowest reached value: \n", np.min(min_values))     

        # if del_beta == del_betas[-1]:
        #     means_1000, stds_1000 = SK_AK(del_beta, beta_init, 1000)
        


if __name__ == '__main__':
    main()





