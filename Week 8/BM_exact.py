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
    #zie notitie voor uitleg
    #checken of hier de fout zit:
    # new_pattern = np.copy(pattern)
    # new_pattern[site] = 1- new_pattern[site]
    # a_orig = (np.exp(0.5*np.dot(new_pattern,np.dot(w,new_pattern)) + np.dot(theta, new_pattern)))/(np.exp(0.5*np.dot(pattern,np.dot(w,pattern)) + np.dot(theta, pattern)))
    a = np.exp((1-2*pattern[site]) * np.dot(w[site], pattern) + (1 - 2 * pattern[site]) * theta[site])
    # print(a - a_orig)
    # diff = (2*pattern[site]-1)*np.dot(w[site], pattern) - (2*pattern[site]-1)*theta[site]
    return a


def MH_sampler(w, theta, NrofFlips, samples, pattern):
    singles = np.empty((samples,len(pattern)))
    doubles = np.empty((samples,len(pattern), len(pattern)))
    for i in range(samples):
        for _ in range(NrofFlips):
            flip = np.random.randint(0,len(pattern))
            a  = a_value(w,theta,pattern, flip)
            if a >= 1:
                pattern[flip] = 1 - pattern[flip]
            elif np.random.random() < a:
                pattern[flip] = 1 - pattern[flip]     
        sample = np.copy(pattern)  
        singles[i] = sample
        w_spins = np.outer(sample, sample)
        doubles[i] = w_spins
    singles_mean = np.mean(singles, axis=0)
    doubles_mean = np.mean(doubles, axis=0)
    np.fill_diagonal(doubles_mean, 0.)
    # print(singles_mean, doubles_mean)
    return singles_mean, doubles_mean, pattern

def MF_and_LR_calculation(w,theta, threshold):
    # first needed to calculate the m's, through some fixed point iteration
    m = np.random.rand(len(theta))
    i = 0 
    while 1: 
        i += 1 
        # print("new : \n", m)
        diff = np.tanh(np.dot(w,m) + theta) - m
        if np.all(np.abs(diff) < threshold):
            m = m + diff
            break
        else:
            m = m + diff
        # print(m)
        
        # m = np.tanh(np.dot(w,m) + theta)
        # i += 1
        # if i % 100 == 0:
        #      print("m_before: ", m)
        # m_new = np.tanh(np.dot(w,m) + theta) #volgensmij klopt dit, maar TODO: dit checken
        # if i % 100 == 0:
        #     print("m_new: " ,m_new)
        #     print("m: ", m)
        # if np.allclose(m_new, m, rtol=0, atol = threshold):
        #     m = m_new
        #     break
        # else:
        #     m = m_new
        # if i % 100 == 0:
        #     print("m_after: ", m)
        #     print("iteration: ", i)
    single_free = np.copy(m)
    A = np.zeros_like(w)
    np.fill_diagonal(A, 1/(1-m**(2)))
    A = A - w
    ksi = np.linalg.inv(A)
    outer_m = np.outer(m,m)
    double_free = ksi + outer_m
    #Again fill diagional with zero to ensure that w does not change on the diagional and thus has an all zero diagonal
    np.fill_diagonal(double_free, 0.)

    return single_free, double_free

def clamped_statistics(data):
    single = 1/(len(data)) * np.sum(data, axis = 0)
    data_needed = np.array([np.outer(x,x) for x in data])
    double = 1/(len(data)) * np.sum(data_needed, axis = 0)
    #Diagonals are set to zero. Since the diagonal of the w should be zero anyway.
    np.fill_diagonal(double, 0.)
    return single, double


def free_statistics(w, theta, all_states, all_states_outer, w_zero = False):
    p_s =  p_s_w(all_states, w, theta)
    single = np.sum([s * p for s, p in zip(all_states, p_s)], axis = 0)
    if w_zero:
        double = 3000
    else:
        double = np.sum([out * p for out, p in zip(all_states_outer, p_s)], axis = 0)
        #Diagonals are set to zero. Since the diagonal of the w should be zero anyway.
        np.fill_diagonal(double, 0.)
    return single, double


def BM_exact(data, NrofSpins, NrofData, lr, threshold, seed=None, w_zero = False, MH_stats = False, samples = 100, sweeps = 1, MF_and_LR = False, MF_threshold = 0.001):
    #initialize w and theta randomly
    if seed:
        np.random.seed(seed)
    if w_zero:
        w = np.zeros((NrofSpins, NrofSpins))
    else:        
        w = np.random.rand(NrofSpins, NrofSpins)
        w = np.tril(w) + np.tril(w, -1).T
        np.fill_diagonal(w,0.) 
    theta = np.random.rand(NrofSpins)


    all_states = list(product(range(2), repeat = NrofSpins))
    all_states = [np.array(s) for s in all_states]

    if MH_stats:
        pattern = np.random.randint(0,2,theta.shape[0])
    else:
        all_states_outer = [np.outer(s,s) for s in all_states]

    likelihood_chain = [likelihood(w, theta, all_states, data)]
    gradient_chain = []
    print(np.mean(w))
    i = 0
    single_clamped, double_clamped = clamped_statistics(data)
    
    while 1:
        if MH_stats:
            single_free, double_free, pattern = MH_sampler(w, theta, NrofSpins * sweeps, samples, pattern)
        elif MF_and_LR:
            single_free, double_free = MF_and_LR_calculation(w,theta, MF_threshold)
        else:
            single_free, double_free = free_statistics(w, theta, all_states, all_states_outer, w_zero)

        gradient_double = double_clamped - double_free
        if w_zero:
            w = np.zeros((NrofSpins, NrofSpins))
        else:
            w += lr * gradient_double
        gradient_single = single_clamped - single_free
        theta += lr * gradient_single

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
            if MH_stats:
                print("ones in pattern:", np.sum(pattern))
            print(i)
            
    print(np.mean(w))
    gradient_chain = np.array(gradient_chain)
    return w, theta, likelihood_chain, gradient_chain


if __name__ == '__main__':
    # Create toy model dataset
    data = np.array([np.random.randint(0, 2, size = args.S) for _ in range(args.N)])
    w, theta, likelihood_chain, gradient_chain = BM_exact(data, args.S, args.N, lr, threshold, False, False)

    plt.plot([x for x in range(len(likelihood_chain))], likelihood_chain)
    plt.xlabel("iterations")
    plt.ylabel("likelihood")
    plt.title("Likelihood over iterations of exact BM for toy model")
    plt.show()
