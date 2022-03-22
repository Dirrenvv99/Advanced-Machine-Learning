import numpy as np
import matplotlib.pyplot as plt
from BM_exact import bm_exact, free_statistics, clamped_statistics
import argparse

parser = argparse.ArgumentParser(description= 'Toy model BM')
parser.add_argument('-N',type= int, default= 20, help='Size of the dataset')
parser.add_argument('-S',type = int, default = 20, help = "Amount of spins" )
parser.add_argument('--eta',type = float, default = 0.005, help = "learningrate" )
parser.add_argument('--threshold',type = float, default = 10**(-13), help = "Threshold for convergence of the method" )
args = parser.parse_args()

lr = args.eta
threshold = args.threshold

def a_value(w, theta, pattern, site):
    diff = np.exp(-pattern[site]*np.dot(w[site], pattern) - 2 * pattern[site] * theta[site])
    return diff


def MH_sampler(data, w, theta, NrofFlips, NrofSamples):
    singles = [] 
    doubles = [] 
    for _ in range(NrofSamples):
        random_patterns = np.random.choice(range(len(data)), size = NrofFlips)
        random_sites = np.random.choice(range(len(data[0])), size = NrofFlips)

        for index, pattern in enumerate(random_patterns):
            a  = a_value(w,theta,data[pattern],random_sites[index])
            if a > 1:
                data[pattern][random_sites[index]] = -1 * data[pattern][random_sites[index]]
            elif np.random.random() < a:
                data[pattern][random_sites[index]] = -1 * data[pattern][random_sites[index]]

        singles.append(np.sum(data, axis = 0))
        doubles.append(np.sum(np.array([np.outer(s,s) for s in data]), axis = 0))
    return np.mean(singles, axis = 0), np.mean(doubles, axis = 0)
    
def comparer(data, NrofFlips):
    # randomly initialize w and theta
    w = np.random.randn(len(data), len(data))
    np.fill_diagonal(w,0.) 
    theta = np.random.randn(len(data))

    condition = True
    single_clamped, double_clamped = clamped_statistics(data,w,theta)

    while condition:
        #exact gradient computation
        single_free, double_free = free_statistics(data,w,theta)
        gradient_double = double_clamped - double_free
        w += lr * gradient_double
        gradient_single = single_clamped - single_free
        theta += lr * gradient_single

        single_free_sampled, double_free_sampled = MH_sampler(data, w, theta, len(data)**2, 3)

        gradient_double_sampled = double_clamped - double_free_sampled
        gradient_single_sampled = single_clamped - single_free_sampled

        
