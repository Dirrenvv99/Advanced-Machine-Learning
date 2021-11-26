import numpy as np
import scipy.special as sps
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm

t = np.loadtxt("t.txt")
x = np.loadtxt("x.txt")
alpha = 0.5


def y(w):
    a = np.dot(x,w)
    return 1/(1+np.exp(-a)) 

def E(w):
    return 1/2*np.dot(w,w)

def G(w):
    return -np.sum(t*np.log(y(w)) + (1-t)*np.log(1-y(w)))

def M(w):
    return G(w) + alpha*E(w)

def p_star(w):
    return np.exp(-M(w))

def sample_q(w_r, sigma):
    return np.random.multivariate_normal(w_r, sigma * np.eye(3), 1)[0]

def q(w,w_r,sigma):
    return multivariate_normal.pdf(w,mean = w_r, cov = sigma * np.eye(3))

def a_value(w_r,w_sample, sigma):

    nom = p_star(w_sample) * q(w_r,w_sample,sigma)
    denom = p_star(w_r) * q(w_sample, w_r, sigma)

    return nom/denom

def MH(sigma, iter):
    w = np.random.multivariate_normal(np.array([0,0,0]), np.eye(3))
    rejections = 0
    sample_list = []
    for _ in range(iter):
        sample = sample_q(w, sigma)
        a = a_value(w,sample, sigma)
        if a >= 1:
            w = sample
        else:
            if np.random.random() < a:
                w = sample
            else: 
                rejections +=1
        sample_list.append(w)
    return sample_list, rejections

def main():
    epochs = 10
    sigma = 10
    w_values, rejections = MH(sigma,10000)
    plt.plot([i for i in range(10000)], [w[0] for w in w_values])
    plt.plot([i for i in range(10000)], [w[1] for w in w_values])
    plt.plot([i for i in range(10000)], [w[2] for w in w_values])
    plt.show()
if __name__ == '__main__':
    main()
