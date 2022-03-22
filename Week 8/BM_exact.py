import numpy as np
import matplotlib.pyplot as plt
import argparse
from itertools import product

parser = argparse.ArgumentParser(description= 'Toy model BM')
parser.add_argument('-N',type= int, default= 10, help='Size of the dataset')
parser.add_argument('-S',type = int, default = 10, help = "Amount of spins" )
parser.add_argument('--eta',type = float, default = 0.5, help = "learningrate" )
parser.add_argument('--threshold',type = float, default = 10**(-13), help = "Threshold for convergence of the method" )
args = parser.parse_args()

lr = args.eta
threshold = args.threshold


def unnormalized_p(s, w,theta):
    return np.exp(0.5*np.dot(s,np.dot(w,s)) + np.dot(theta, s))


def partition(w,theta, all_states):
    return np.sum([unnormalized_p(np.array(s),w,theta) for s in all_states])

def p_s_w(s,w, theta, partition):
    # for the exact model needs to be calculated exactly, thus including the normalization constant. 
    return 1/partition * unnormalized_p(s,w,theta)


def likelihood(w,theta, all_states):
    nom = 0 
    Z = partition(w,theta, all_states)
    for s in data:
        nom += 0.5*(np.dot(s,np.dot(w,s))) + np.dot(theta, s)
    return nom/len(data) - np.log(Z)

def clamped_statistics(data):
    single = 1/(len(data)) * np.sum(data, axis = 0)
    data_needed = np.array([np.outer(x,x) for x in data])
    double = 1/(len(data)) * np.sum(data_needed, axis = 0)

    return single, double


def free_statistics(w, theta, all_states):
    Z = partition(w,theta, all_states)
    single = np.sum([np.array(s) * p_s_w(np.array(s), w, theta, Z) for s in all_states], axis = 0)
    double = np.sum([np.outer(np.array(s),np.array(s)) * p_s_w(np.array(s), w, theta, Z) for s in all_states], axis = 0)
    return single, double


def BM_exact(data, NrofSpins, NrofData):
    #Create toy model dataset


    #initialize w and theta randomly
    w = np.random.randn(NrofSpins, NrofSpins)
    np.fill_diagonal(w,0.) 
    theta = np.random.randn(NrofSpins)

    condition = True
    all_states = list(product(range(2), repeat = w.shape[0]))
    likelihood_chain = [likelihood(w,theta, all_states)]
    gradient_chain = []
    i = 0
    single_clamped, double_clamped = clamped_statistics(data)
    
    while condition:
        single_free, double_free = free_statistics(w,theta, all_states)
        gradient_double = double_clamped - double_free
        w += lr * gradient_double
        gradient_single = single_clamped - single_free
        theta += lr * gradient_single

        likelihood_chain.append(likelihood(w,theta, all_states))
        gradient_chain.append((np.mean(gradient_single), np.mean(gradient_double)))

        if np.allclose(double_clamped, double_free, atol = threshold * 1/lr) and np.allclose(single_clamped, single_free, atol = threshold * 1/lr):
            condition = False
        
        i += 1
        # if i % 20 == 0:
        #     print("double: ", (double_clamped - double_free))
        #     print("single: ", (single_clamped - single_free))
        if i % 100 == 0:
            print("double: ",np.mean(np.abs(double_clamped - double_free)))
            print("single: ",np.mean(np.abs(single_clamped - single_free)))
            print(i)
    return w, theta, likelihood_chain, gradient_chain



if __name__ == '__main__':
    data = np.array([np.random.randint(0,2,size = args.S) for _ in range(args.N)])
    for index, point in enumerate(data):
        point[point==0] = -1
        data[index] = point
    w, theta, likelihood_chain, gradient_chain = BM_exact(data, len(data), 10)

    plt.plot([x for x in range(len(likelihood_chain))], likelihood_chain)
    plt.xlabel("iterations")
    plt.ylabel("likelihood")
    plt.title("Likelihood over iterations of exact BM for toy model")
    plt.show()
    






    




