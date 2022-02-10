'''Performs the data analysis for the report based on the JSON files created by the other codes'''

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import json
import argparse
from sys import platform 
import os


parser = argparse.ArgumentParser(description= 'Data analysis for the opitimization methods used  for Ising model energy.')
parser.add_argument('--method', default= "NOT_FOUND", help='AK : AK schedule; EXP: exponential schedule; ITERATIVE_500: w500 problem, iterative method; ITERATIVE_FRUSTRATED: random frsutrated problem, iterative method; ITERATIVE_FERRO: ferromagnetic problem, iterative method')
args = parser.parse_args()


if platform == "linux" or platform == "linux2" or platform == "darwin":
    SA_AK_DIR = "./SA_AK"
    SA_EXP_DIR = "./SA_EXP"
    IT_500_DIR = "./ITERATIVE_500"
    IT_FERRO_DIR = "./ITERATIVE_FERRO"
    IT_FRUSTRATED_DIR = "./ITERATIVE_FRUSTRATED"
elif platform == "win32":
    SA_AK_DIR = ".\SA_AK"
    SA_EXP_DIR = ".\SA_EXP"
    IT_500_DIR = ".\ITERATIVE_500"
    IT_FERRO_DIR = ".\ITERATIVE_FERRO"
    IT_FRUSTRATED_DIR = ".\ITERATIVE_FRUSTRATED"

if args.schedule == "AK":
    for file in os.scandir(SA_AK_DIR):
        if file.is_file():
            with open(file.path) as f:                
                data_dict = json.load(f)
                del_beta = data_dict['del_beta:']
                L = data_dict['L']
                fig, axs = plt.subplots(1,3, constrained_layout = True)
                fig.set_size_inches((20, 11), forward=False)
                axs[0].plot([x for x in range(len(data_dict['means']))], data_dict['means'], label = "mean values for " + str(del_beta))
                axs[0].legend()
                axs[0].set_xlabel("#chain")
                axs[0].set_ylabel("Mean energy value")
                axs[0].set_title("Mean value over chains")
                axs[1].plot([x for x in range(len(data_dict['stds: ']))], data_dict['stds: '], label = "stds values for " + str(del_beta))
                axs[1].legend()
                axs[1].set_xlabel("#chain")
                axs[1].set_ylabel("Std of Energy")
                axs[1].set_title("Std of Energy over the chains")
                axs[2].plot([x for x in range(len(data_dict['betas:']))], data_dict['betas:'], label = "beta values for " + str(del_beta))
                axs[2].legend()
                axs[2].set_xlabel("#chain")
                axs[2].set_ylabel("beta")
                axs[2].set_title("Evolution of beta over chains")
                plt.suptitle("Delta Beta: " + str(del_beta) + " Simulated Annealing with AK schedule")
                # plt.savefig(str(del_beta) + "SA_AK_SC.png", bbox_inches = "tight")
                fig.savefig(str(del_beta) + "_SA_AK_SC.png", dpi=500)
elif args.schedule == "EXP":
    for file in os.scandir(SA_EXP_DIR):
        if file.is_file():
            with open(file.path) as file:                
                data_dict = json.load(file)
                f = data_dict['f:']
                L = data_dict['L']
                fig, axs = plt.subplots(1,3, constrained_layout = True)
                fig.set_size_inches((20, 11), forward=False)
                axs[0].plot([x for x in range(len(data_dict['means']))], data_dict['means'], label = "mean values for " + str(f))
                axs[0].legend()
                axs[0].set_xlabel("#chain")
                axs[0].set_ylabel("Mean energy value")
                axs[0].set_title("Mean value over chains")
                axs[1].plot([x for x in range(len(data_dict['stds: ']))], data_dict['stds: '], label = "stds values for " + str(f))
                axs[1].legend()
                axs[1].set_xlabel("#chain")
                axs[1].set_ylabel("Std of Energy")
                axs[1].set_title("Std of Energy over the chains")
                axs[2].plot([x for x in range(len(data_dict['betas:']))], data_dict['betas:'], label = "beta values for " + str(f))
                axs[2].legend()
                axs[2].set_xlabel("#chain")
                axs[2].set_ylabel("beta")
                axs[2].set_title("Evolution of beta over chains")
                plt.suptitle("f : " + str(f) + " Simulated Annealing with exponential schedule")
                # plt.savefig(str(del_beta) + "SA_AK_SC.png", bbox_inches = "tight")
                fig.savefig(str(f) + "_SA_EXP_SC.png", dpi=500)
elif args.schedule == "ITERATIVE_500":
    for file in os.scandir(IT_500_DIR):
        if file.is_file():
            with open(file.path) as file:
                data_dict = json.load(file)
                K = data_dict['K']
                L = data_dict['L']
                energy = np.array(data_dict['energy'])
                fig = plt.figure()
                plt.plot([x for x in range(len(energy))], energy)
                fig.set_size_inches((11,11), forward = False)
                plt.xlabel("#iterations")
                plt.ylabel("Energy")
                plt.title("Energy of solution for 1 loop/restart, with K = " + str(K) + ", for W500 problem")
                plt.savefig("K_" + str(K) + "_Energy_evolution_it_500.png", dpi = 500)
elif args.schedule == "ITERATIVE_FERRO":
    for file in os.scandir(IT_FERRO_DIR):
        if file.is_file():
            with open(file.path) as file:
                data_dict = json.load(file)
                K = data_dict['K']
                L = data_dict['L']
                energy = np.array(data_dict['energy'])
                fig = plt.figure()
                plt.plot([x for x in range(len(energy))], energy)
                fig.set_size_inches((11,11), forward = False)
                plt.xlabel("#iterations")
                plt.ylabel("Energy")
                plt.title("Energy of solution for 1 loop/restart, with K = " + str(K) + ", for random ferromagnetic problem")
                plt.savefig("K_" + str(K) + "_Energy_evolution_it_ferro.png", dpi = 500)
elif args.schedule == "ITERATIVE_FRUSTRATED":
    for file in os.scandir(IT_FRUSTRATED_DIR):
        if file.is_file():
            with open(file.path) as file:
                data_dict = json.load(file)
                K = data_dict['K']
                L = data_dict['L']
                energy = np.array(data_dict['energy'])
                fig = plt.figure()
                plt.plot([x for x in range(len(energy))], energy)
                fig.set_size_inches((11,11), forward = False)
                plt.xlabel("#iterations")
                plt.ylabel("Energy")
                plt.title("Energy of solution for 1 loop/restart, with K = " + str(K) + ", for random frustrated problem")
                plt.savefig("K_" + str(K) + "_Energy_evolution_it_frustrated.png", dpi = 500)
elif args.schedule == "NOT_FOUND":
    print("The wanted data analysis is not recognized!")

        
        



        


            




