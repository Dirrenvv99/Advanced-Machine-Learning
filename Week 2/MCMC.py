import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm
import matplotlib.pyplot as plt

def p_star(x):
    return multivariate_normal.pdf(x, mean = [0,0], cov = [[250.25,-249.75],[-249.75,250.25]])

def sample_q(x_r, sigma):
    return np.random.multivariate_normal(x_r, sigma * np.eye(2), 1)[0]

def q(x,x_r,sigma):
    return multivariate_normal.pdf(x,mean = x_r, cov = sigma * np.eye(2))

def a_value(x_r, x_sample, sigma):

    nom = p_star(x_sample) * q(x_r,x_sample,sigma)
    denom = p_star(x_r) * q(x_sample, x_r, sigma)

    return nom/denom

def MH(sigma, iter):
    x = np.random.multivariate_normal(np.array([0,0]), np.eye(2))
    rejections = 0
    for _ in range(iter):
        sample = sample_q(x, sigma)
        a = a_value(x, sample, sigma)
        if a >= 1:
            x = sample
        else:
            if np.random.random() < a:
                x = sample
            else: 
                rejections +=1
    return x, rejections

def main():
    sigmas = np.linspace(0.2,3,10)
    mean_values = []
    rejection_rates = []
    for sigma in tqdm(sigmas):
        samples = []
        recs = []
        for _ in tqdm(range(100)):
            x_final, rejections = MH(sigma, 100)
            samples.append(x_final)
            recs.append(rejections)
        mean_values.append(np.mean(samples))
        rejection_rates.append(np.mean(rejections)/100)
    plt.plot(sigmas, rejection_rates, label = "rejection per sigma")
    plt.plot(sigmas, mean_values, label = "mean per sigma")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
