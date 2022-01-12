import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from makedata_python import make_data
import argparse
from tabulate import tabulate
import time

# w = make_data(5,False)
w = np.loadtxt("w500")
beta_init_AK = 0.001
beta_init_exp = 0.1
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
        if a >= 1:
            x[site] = -1 * x[site]
            energy = energy + diff
        else:
            if np.random.random() < a:
                x[site] = -1*x[site]
                energy = energy + diff
        energies.append(energy)
    return x, energies
    
def determine_initial_beta(x):
    #Notice that this implementation just tries all different spin flip possibilities, which is practically the same as proposed within the slides
    highest_diff = -1 * np.inf
    for site in range(0,len(x)):
        diff = E_dif(x,site)
        if diff > highest_diff:
            highest_diff = diff
    return 1/highest_diff

def SK(L, AK=True, del_beta=0, f=0):
    var_E = np.inf
    means = []
    x = np.random.randint(0,2,size = w.shape[0])
    x[x == 0] = -1 
    energy = E(x)
    beta = determine_initial_beta(x)
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
        x, energies = MH(x,beta,sites, energy)        
        energy = energies[-1]
        means.append(np.mean(energies))
        var_E = np.std(energies)
        vars.append(var_E)
    
    # print("\n", counter)
    # plt.plot([x for x in range(counter)],betas)
    # plt.show()
    return means

def main():
    if exp_schedule == False:
        del_betas = [0.1, 0.01, 0.001]
        fs = [1.01, 1.001, 1.0002]
        table = []
        L = 500
        if exp_schedule:
            enum = fs
        else: 
            enum = del_betas
        for index, del_beta in enumerate(enum):
            if index == (len(del_betas) - 1):
                L = 1000
            min_values = []
            running_times = []
            for _ in tqdm(range(N_runs)):
                start_time = time.time()
                if exp_schedule:
                    means = SK(L, AK=True, del_beta = del_betas)
                else:
                    means = SK(L, AK=False, f = fs)
                stop_time = time.time()
                running_time = round(stop_time - start_time, 1)
                min_values.append(means[-1])
                running_times.append(running_time)
            table.append((del_beta, L, np.min(running_times), str(np.mean(min_values)) + " +- " + str(round(np.std(min_values), 0))))      
            print("minimal cost: \n", np.mean(min_values), " +- ", np.std(min_values))
            print("lowest reached value: \n", np.min(min_values))
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





