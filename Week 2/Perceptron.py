import numpy as np
import scipy.special as sps
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm

t = np.loadtxt("t.txt")
x = np.loadtxt("x.txt")
alpha = 0.01


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
    return -M(w)

def sample_q(w_r, sigma):
    return np.random.multivariate_normal(w_r, sigma * np.eye(3)).tolist()

def q(w,w_r,sigma):
    return multivariate_normal.logpdf(w,mean = w_r, cov = sigma * np.eye(3)).tolist()

def a_value(w_r,w_sample, sigma):

    nom = p_star(w_sample) + q(w_r,w_sample,sigma)
    denom = p_star(w_r) + q(w_sample, w_r, sigma)

    return np.exp(nom-denom)

def MH(sigma, iter):
    w = np.random.multivariate_normal([0,0,0], np.eye(3)).tolist()
    rejections = 0
    sample_list = []
    a_values = []
    for i in tqdm(range(iter)):
        sample = sample_q(w, sigma)
        a = a_value(w,sample, sigma)
        a_values.append(a)
        if a >= 1:
            w = sample
        else:
            if np.random.random() < a:
                w = sample
            else: 
                rejections +=1
        sample_list.append(w)
    return sample_list, rejections, a_values

def main():
    np.random.seed(42)
    epochs = 10
    sigma = 0.1
    N = 100000
    w_values, rejections, a_values = MH(sigma,N)
    w_values = np.array(w_values)
    # to show that it works:
    w_samples_indices =  np.random.choice(np.arange(0, len(w_values)), replace=False, size= 1000)
    y_samples = [y(w_values[i]) for i in w_samples_indices]
    y_mean = np.mean(y_samples, axis = 0)
    mean = np.mean([M(w) for w in w_values])
    print('burn_in:' , next(w[0] for w in enumerate(w_values) if M(w[1]) < mean))
    print('mean:', mean)
    print("rejections: ",rejections/N)
    
    fig, axs = plt.subplots(4)

    axs[0].plot([i for i in range(N)], [w[0] for w in w_values], label = "w0")
    axs[0].plot([i for i in range(N)], [w[1] for w in w_values], label = "w1")
    axs[0].plot([i for i in range(N)], [w[2] for w in w_values], label = "w2")
    axs[0].set_ylabel('Weight value')
    axs[0].set_xlabel('Number of iterations')
    axs[0].set_title('weight values over time')
    axs[0].legend()

    axs[1].scatter([w[2] for w in w_values], [w[1] for w in w_values])
    axs[1].set_ylim([-10,10])
    axs[1].set_xlim([-10,10])
    axs[1].set_xlabel('w2 value')
    axs[1].set_ylabel('w1 value')
    axs[1].set_title('w1 vs w2')


    axs[2].plot([i for i in range(N)], [G(w) for w in w_values])
    axs[2].set_ylim([0,14])
    axs[2].set_xlabel('Number of iterations')
    axs[2].set_ylabel('G(W)')
    axs[2].set_title('G(w) over time')

    axs[3].plot([i for i in range(N)], [M(w) for w in w_values])
    axs[3].set_ylim([0,14])
    axs[3].set_xlabel('Number of iterations')
    axs[3].set_ylabel('M(W)')
    axs[3].set_title('M(w) over time')

    fig.tight_layout(pad = 0.5)
    plt.show()
    plt.hist(a_values, bins = [x/1000 for x in range(2000)])
    plt.show()
if __name__ == '__main__':
    main()
