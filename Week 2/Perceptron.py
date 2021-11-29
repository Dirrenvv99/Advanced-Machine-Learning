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
    result = 0
    y_list = y(w)
    for i in range(len(y_list)):
        if y_list[i] == 1:
            result += t[i]*np.log(y_list[i]) + (1 - t[i]) * np.log(1-y_list[i] + 10**(-77))
        else:
            if y_list[i] == 0:
                result += t[i]*np.log(y_list[i] + 10**(-77)) + (1 - t[i]) * np.log(1-y_list[i])
            else:
                result += t[i]*np.log(y_list[i]) + (1 - t[i]) * np.log(1-y_list[i])
    return -1 * result

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
    for _ in tqdm(range(iter)):
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
    N = 40000
    w_values, rejections = MH(sigma,N)
    print(y(w_values[-1]))
    fig, axs = plt.subplots(4)

    axs[0].plot([i for i in range(N)], [w[0] for w in w_values], label = "w0")
    axs[0].plot([i for i in range(N)], [w[1] for w in w_values], label = "w1")
    axs[0].plot([i for i in range(N)], [w[2] for w in w_values], label = "w2")
    axs[0].legend()

    axs[1].scatter([w[0] for w in w_values], [w[1] for w in w_values])

    axs[2].plot([i for i in range(N)], [G(w) for w in w_values])
    axs[2].set_ylim([0,14])

    axs[3].plot([i for i in range(N)], [M(w) for w in w_values])
    axs[3].set_ylim([0,14])

    fig.tight_layout(pad = 1.5)
    plt.show()
if __name__ == '__main__':
    main()
