import numpy as np
import matplotlib.pyplot as plt
import argparse
from itertools import product

parser = argparse.ArgumentParser(description= 'Toy model BM')
parser.add_argument('-N',type= int, default= 1000, help='Size of the dataset')
parser.add_argument('-S',type = int, default = 10, help = "Amount of spins" )
parser.add_argument('--eta',type = float, default = 0.5, help = "learningrate" )
parser.add_argument('--threshold',type = float, default = 10**(-13), help = "Threshold for convergence of the method" )
args = parser.parse_args()

lr = args.eta
threshold = args.threshold


def unnormalized_p(all_states, w,theta):
    return np.array([np.exp(0.5*np.dot(s, np.dot(w,s)) + np.dot(theta, s)) for s in all_states])


def p_s_w(all_states, w, theta):
    # for the exact model needs to be calculated exactly, thus including the normalization constant. 
    res = unnormalized_p(all_states,w,theta)
    return 1/np.sum(res) * res


def likelihood(w,theta, all_states, data):
    Z = np.sum(unnormalized_p(all_states,w,theta))
    nom = [0.5*np.dot(s,np.dot(w,s)) + np.dot(theta, s) for s in data] 
    return np.mean(nom, axis=0) - np.log(Z)

def a_value(w, theta, pattern, site):
    diff = np.exp((2*pattern[site]-1)*np.dot(w[site], pattern) - (2*pattern[site]-1)*theta[site])
    return diff


def MH_sampler(w, theta, NrofFlips, iters):
    singles = [] 
    doubles = []
    pattern = np.random.randint(0,2,NrofFlips)
    for _ in range(iters):
        flip = np.random.randint(0, NrofFlips)
        a  = a_value(w,theta,pattern, flip)
        if a > 1:
            pattern[flip] = -(pattern[flip]-1)
            singles.append(pattern)
            w_spins = np.outer(pattern, pattern)
            np.fill_diagonal(w_spins, 0)
            doubles.append(w_spins)
        elif np.random.random() < a:
            pattern[flip] = -(pattern[flip]-1)
            singles.append(pattern)
            w_spins = np.outer(pattern, pattern)
            np.fill_diagonal(w_spins, 0)
            doubles.append(w_spins)
    singles_mean = np.mean(singles, axis=0)
    doubles_mean = np.mean(doubles, axis=0)
    # print(singles_mean, doubles_mean)
    return singles_mean, doubles_mean


def clamped_statistics(data):
    single = 1/(len(data)) * np.sum(data, axis = 0)
    data_needed = np.array([np.outer(x,x) for x in data])
    double = 1/(len(data)) * np.sum(data_needed, axis = 0)
    np.fill_diagonal(double, 0)
    return single, double


def free_statistics(w, theta, all_states, all_states_outer, w_zero = False):
    p_s =  p_s_w(all_states, w, theta)
    single = np.sum([s * p for s, p in zip(all_states, p_s)], axis = 0)
    if w_zero:
        double = 3000
    else:
        double = np.sum([out * p for out, p in zip(all_states_outer, p_s)], axis = 0)
        np.fill_diagonal(double, 0)
    return single, double


def BM_exact(data, NrofSpins, NrofData, lr, threshold, w_zero = False, MH_stats = False):
    #initialize w and theta randomly
    if w_zero:
        w = np.zeros((NrofSpins, NrofSpins))
    else:
        w = np.random.randn(NrofSpins, NrofSpins)
        np.fill_diagonal(w,0.) 
    theta = np.random.randn(NrofSpins)

    if (not MH_stats):
        all_states = list(product(range(2), repeat = NrofSpins))
        all_states = [np.array(s) for s in all_states]
        all_states_outer = [np.outer(s,s) for s in all_states]
        likelihood_chain = [likelihood(w, theta, all_states, data)]
        gradient_chain = []
    print(np.mean(w))
    i = 0
    single_clamped, double_clamped = clamped_statistics(data)
    
    while 1:
        if (not MH_stats):
            single_free, double_free = free_statistics(w, theta, all_states, all_states_outer, w_zero)
        else:
            single_free, double_free = MH_sampler(w, theta, NrofSpins, 1000)
        gradient_double = double_clamped - double_free
        if w_zero:
            w = np.zeros((NrofSpins, NrofSpins))
        else:
            w += lr * gradient_double
            np.fill_diagonal(w, 0.)
        gradient_single = single_clamped - single_free
        theta += lr * gradient_single

        if (not MH_stats):
            likelihood_chain.append(likelihood(w,theta, all_states, data))
            gradient_chain.append((np.mean(gradient_single), np.mean(gradient_double)))
        
        if w_zero:
           if np.allclose(single_clamped, single_free, rtol=0, atol = threshold * 1/lr):
               break
        else:
            if np.allclose(double_clamped, double_free, rtol=0, atol = threshold * 1/lr) and \
            np.allclose(single_clamped, single_free, rtol=0, atol = threshold * 1/lr):
                break
        
        i += 1
        if i % 100 == 0:
            print("double: ",np.max(np.abs(double_clamped - double_free)))
            print("single: ",np.max(np.abs(single_clamped - single_free)))
            print(i)
            
    print(np.mean(w))
    return w, theta, likelihood_chain, gradient_chain


if __name__ == '__main__':
    # Create toy model dataset
    data = np.array([np.random.randint(0, 2, size = args.S) for _ in range(args.N)])
    w, theta, likelihood_chain, gradient_chain = BM_exact(data, args.S, args.N, lr, threshold)

    plt.plot([x for x in range(len(likelihood_chain))], likelihood_chain)
    plt.xlabel("iterations")
    plt.ylabel("likelihood")
    plt.title("Likelihood over iterations of exact BM for toy model")
    plt.show()
