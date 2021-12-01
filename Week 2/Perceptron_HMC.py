import numpy as np
import scipy.special as sps
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm

t = np.loadtxt("t.txt")
x = np.loadtxt("x.txt")
alpha = 0.5

def sigmoid(x):
    return 1/(1+np.exp(-x))

def y(w):
    a = np.dot(x,w)
    return sigmoid(a)

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

def grad_G(w):
    result = np.array([0,0,0])
    y_list = y(w)
    for i in range(len(y_list)):
        result += -t[i]/y_list(w) - (1-t[i])/(1-y_list[i]) * y_list[i] * (1 - y_list[i]) * x[i]
    return result 

def grad_E(w):
    return w

def grad_M(w):
    return grad_G(w) + alpha * grad_E(w)

def HMC(tau, eps, iters = 100):
    w = np.random.multivariate_normal(np.array([0,0,0]), np.eye(3), 1)[0]
    w_values = []
    for _ in range(iters):
        p = np.random.multivariate_normal(np.array([0,0,0]), np.eye(3), 1)[0]

        H_old = np.dot(p,p)/2 + M(w)
        w_old = np.copy(w)

        for _ in range(tau):
            p -= eps * grad_M(w)/2
            w += eps *p
            p -= eps * grad_M(w)/2

        H_new = np.dot(p,p)/2 + M(w)
        dH = H_new - H_old

        if dH < 0 or np.random.random() < np.exp(-dH):
            w = w
        else:
            w = w_old
        w_values.append(w)
    return w_values

def main():
    tau = 100
    eps = 0.015
    iters = 100

    w_values = HMC(tau, eps, iters)

if __name__ == "__main__":
    main()


