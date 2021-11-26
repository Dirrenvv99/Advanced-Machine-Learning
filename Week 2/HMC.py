import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def E(x):
    return 0.5*(250.25*x[0]**2 - x[0]*x[1]*2*249.75 + 250.25*x[1])


def gradE(x):
    return np.array([250.25* x[0] - 249.75*x[1], 250.25* x[1] - 249.75*x[0]])


def update_step(x_old, tau, eps):
    p = np.random.multivariate_normal(np.array([0,0]), np.eye(2), 1)[0] 
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

    if dH < 0 or np.random.random() < np.exp(-1 * dH):
        return x_new, 0

    return x_old, 1


def main():
    iter = 10000
    update_steps = 10

    taus = [10, 30, 50]
    epss = [i*0.001 for i in range(1,10,2)]

    for tau, eps in [(i, j) for i in taus for j in epss]:
        np.random.seed(0)
        samples = []
        rejections = 0
        for _ in tqdm(range(iter)):
            x = np.random.multivariate_normal(np.array([0,0]), np.eye(2), 1)[0]
            for _ in range(update_steps):
                x, rejected = update_step(x, tau, eps)
                rejections += rejected
            samples.append(x)
        
        print(f"{tau},\t{eps}\t{rejections}")
        

if __name__ == '__main__':
    main()