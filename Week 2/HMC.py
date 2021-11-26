import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm
import matplotlib.pyplot as plt
def E(x):
    return 0.5*(250.25*x[0]**2 - x[0]*x[1]*2*249.75 + 250.25*x[1])

def gradE(x):
    return np.array([250.25* x[0] - 249.75*x[1], 250.25* x[1] - 249.75*x[0]])

def update_step(x_old, tau, eps):
    p = np.random.multivariate_normal(0, np.eye(2), 1)[0] 
    H_old = np.dot(p,p)/2 + E(x_old)
    grad = gradE(x_old)

    for _ in range(tau):
        p = p - eps * grad/2
        x_new  = x_old + eps * p
        grad = gradE(x_new)
        p = p - eps * grad/2
    
    Enew = E(x_new)
    Hnew = np.dot(p,p)/2 + Enew
    dH = Hnew - H_old

    accept = False
    if dH < 0:
        accept = True
    elif np.random.random() < np.exp(-1 * dH):
        accept = True
 
    if accept:
        return x_new, 0
    else:
        return x_old, 1


def main(iter, update_steps):
    samples = []
    eps = 0.01
    tau = 30
    rejections = 0
    for _ in range(iter):
        x = np.random.multivariate_normal(0, np.eye(2), 1)[0]
        for _ in range(update_steps):
            x, rejected = update_step(x, tau, eps)
            rejections += rejected
        samples.append(x)
        



if __name__ == '__main__':
    main()