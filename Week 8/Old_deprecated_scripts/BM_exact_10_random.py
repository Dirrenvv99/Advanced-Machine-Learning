import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from BM_exact import BM_exact
from itertools import product
from collections import Counter
from sys import platform 
import os

parser = argparse.ArgumentParser(description= 'Toy model BM')
parser.add_argument('-N',type= int, default= 10, help='Size of the dataset; amount of random spins used')
parser.add_argument('-ones',type= int, default= 350, help='Amount of ones desired within the data for computational speed')
parser.add_argument('--eta',type = float, default = 0.001, help = "learningrate" )
parser.add_argument('--threshold',type = float, default = 10**(-5), help = "Threshold for convergence of the method (chosen lower due to much of only zeros)" )
args = parser.parse_args()

lr = args.eta
threshold = args.threshold

def unnormalized_p(s, w,theta):
    return np.exp(0.5*np.dot(s,np.dot(w,s)) + np.dot(theta, s))

def p_s_w(s ,w, theta, Z):
    # for the exact model needs to be calculated exactly, thus including the normalization constant.
    return 1/Z * unnormalized_p(s,w,theta)

def main():
    #Create dataset consisting of 10 random lines of the given data
    full_data = np.loadtxt("bint.txt")
    data_before = full_data[:,:953]

    #seed to make sure it can be recreated
    np.random.seed(42)
    indices_10 = np.random.choice(range(160), size = args.N, replace = False)

    data = data_before[indices_10]
    total_1 = np.sum(data)
    print(total_1)

    #Uncomment below to make sure that data at least contains a little of   
    # while total_1 < args.ones:
    #     print("Data is beign redrawn to avoid too many spikes with only zeros, for the sake of computation speed")
    #     data = data_before[np.random.choice(range(160), size = args.N, replace = False)]
    #     total_1 = np.sum(data)
    data = data.transpose()

    w, theta, likelihood_chain, gradient_chain = BM_exact(data,len(data[0]), len(data), lr, threshold)

    _, theta_indep, _, _ = BM_exact(data, len(data[0]), len(data), lr, threshold, True)

    all_states = list(product(range(2), repeat = len(data[0])))

    full_data = full_data[indices_10]
    full_data = full_data.transpose()

    full_data = full_data.tolist()
    new_data = map(tuple, full_data)
    new_data_set = [list(item) for item in set(tuple(row) for row in full_data)]

    observed_occ = Counter(new_data)

    observed_rate = []
    for i in new_data_set:
        observed_rate.append(observed_occ[tuple(i)]/len(full_data))

    Z = np.sum([unnormalized_p(np.array(s),w,theta) for s in all_states])

    Z_zeros = np.sum([unnormalized_p(np.array(s), np.zeros_like(w),theta_indep) for s in all_states])
    
    approximated_rate = [p_s_w(np.array(s), w, theta, Z) for s in new_data_set]

    approximated_rate_indep = [p_s_w(np.array(s), np.zeros_like(w), theta_indep, Z_zeros) for s in new_data_set]

    plt.scatter(observed_rate, approximated_rate, label = "patterns_dep", color = "red")
    plt.scatter(observed_rate, approximated_rate_indep, label = "patterns_indep", color = "green")        
    plt.plot([x for x in np.linspace(0.000001,1, 10000)],[x for x in np.linspace(0.000001,1, 10000)], color = "red", label = "y = x")
    plt.xlabel("Observerd pattern rate")
    plt.ylabel("Approximated by BM pattern rate")
    plt.xscale('log')
    plt.legend()
    plt.yscale('log')
    plt.title("Recreation fig 2a")

    if platform == "linux" or platform == "linux2" or platform == "darwin":
        plt.savefig("./PLOTS/BM_EXACT_SALAMANDER_" + str(threshold) + "_full.png")
    elif platform == "win32":
        plt.savefig(".\\PLOTS\\BM_EXACT_SALAMANDER_" + str(threshold) + "_full.png")
    plt.show()





        
        


        # fig, axs = plt.subplots(1,2)

        # im = axs[0].imshow(w, cmap = 'bwr')
        # divider = make_axes_locatable(axs[0])
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # axs[0].set_title("Connectivity between neurons")

        # fig.colorbar(im, cax=cax, orientation='vertical')

        # theta = np.array([theta for _ in range(120)])
        # im_theta = axs[1].imshow(theta, cmap = 'bwr')
        # divider_theta = make_axes_locatable(axs[1])
        # cax_theta = divider_theta.append_axes('right', size='5%', pad=0.05)
        # axs[1].set_title("local fields")

        # fig.colorbar(im_theta, cax=cax_theta, orientation='vertical')
        # plt.savefig(".\PLOTS\BM_EXACT_10_RANDOM_NEEEDED_PLOTS.png")
        # plt.show()


if __name__ == '__main__':
    main()
    



        # w = np.random.randn(data.shape[1], data.shape[1])
        # np.fill_diagonal(w,0.) 
        # theta = np.random.randn(data.shape[1])
        # print(w)

        # condition = True
        # likelihood_chain = [likelihood(data,w,theta)]
        # i = 0
        # single_clamped, double_clamped = clamped_statistics(data)
        # print(double_clamped)
        # while condition:
        #     single_free, double_free = free_statistics(data,w,theta)
        #     w += lr * (double_clamped - double_free)
        #     theta += lr * (single_clamped - single_free)
        #     likelihood_chain.append(likelihood(data,w,theta))

        #     if np.allclose(double_clamped, double_free, atol = threshold) and np.allclose(single_clamped, single_free, atol = threshold):
        #         condition = False
            
        #     i += 1
        #     # if i % 20 == 0:
        #     #     print("double: ", (double_clamped - double_free))
        #     #     print("single: ", (single_clamped - single_free))
        #     if i % 100 == 0:
        #         print("double: ",np.mean(np.abs(double_clamped - double_free)))
        #         print("single: ",np.mean(np.abs(single_clamped - single_free)))
        #         print(i)






    




