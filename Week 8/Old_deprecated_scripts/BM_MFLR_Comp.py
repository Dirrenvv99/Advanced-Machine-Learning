import numpy as np
import matplotlib.pyplot as plt
from BM_exact import BM_exact
import argparse

parser = argparse.ArgumentParser(description= 'Toy model BM')
parser.add_argument('-N',type= int, default= 1000, help='Size of the dataset')
parser.add_argument('-S',type = int, default = 10, help = "Amount of spins" )
parser.add_argument('--eta',type = float, default = 0.01, help = "learningrate" )
parser.add_argument('--threshold',type = float, default = 10**(-4), help = "Threshold for convergence of the method" )
args = parser.parse_args()

lr = args.eta
threshold = args.threshold
    
def comparer():
    # generate data for toy model
    data = np.array([np.random.randint(0, 2, size = args.S) for _ in range(args.N)])

    w_MFLR, theta_MFLR, l_MFLR, g_MFLR = BM_exact(data, args.S, args.N,lr,threshold, seed = 42, MF_and_LR= True, MF_threshold= 0.01)

    w_exact, theta_exact, l_exact, g_exact = BM_exact(data, args.S, args.N,lr,threshold, seed = 42)


    fig, axs = plt.subplots(2,2)
    axs[0,0].plot([x for x in range(len(l_exact))], l_exact)
    axs[0,0].set_title("Exact Likelihood")

    axs[0,1].plot([x for x in range(len(l_MFLR))], l_MFLR)
    axs[0,1].set_title("MFLR Likelihood")

    axs[1,0].plot([x for x in range(len(g_exact[:,0]))], g_exact[:,0], label = "single")
    axs[1,0].plot([x for x in range(len(g_exact[:,1]))], g_exact[:,1], label = "double")
    axs[1,0].legend()
    axs[1,0].set_title("Exact mean of gradients")

    axs[1,1].plot([x for x in range(len(g_MFLR[:,0]))], g_MFLR[:,0], label = "single")
    axs[1,1].plot([x for x in range(len(g_MFLR[:,1]))], g_MFLR[:,1], label = "double")
    axs[1,1].legend()
    axs[1,1].set_title("MFLR mean of gradients")

    plt.show()

if __name__ == '__main__':
    comparer()

        
