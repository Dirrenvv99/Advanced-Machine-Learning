import numpy as np
import scipy.special as sps
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm

t = np.loadtxt("t.txt")
x = np.loadtxt("x.txt")
alpha = 0.01

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
    result = np.array([.0,.0,.0])
    y_list = y(w)
    for i in range(len(y_list)):
        result += (t[i] * (1-y_list[i])  - (1-t[i]) * y_list[i])  * x[i]
    return -1*result 

def grad_E(w):
    return w #Dit klopt wel gewoon, want als je afleid naar w_i krijg je w_i terug. Dus in zn geheel volgt w

def grad_M(w):
    return grad_G(w) + alpha * grad_E(w)

def HMC(tau, eps, iters = 100):
    # w = np.random.multivariate_normal(np.array([0,0,0]), np.eye(3))
    w = np.array([.0,.0,.0])
    w_values = []
    rejections = 0
    for _ in tqdm(range(iters)):
        p = np.random.multivariate_normal(np.array([0,0,0]), np.eye(3))

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
            rejections += 1
        w_values.append(w)
    return w_values, rejections

def main():
    eps = np.sqrt(0.02)
    tau = round(1/eps) #Deze waarde kies ik omdat ik van het vak monte carlo van natuurkunde weet dat eps*tau ongeveer 1 moet zijn.
    print(tau)
    iters = 40000

    w_values, rejections = HMC(tau, eps, iters)
    #Deze is wel vrij slecht, maar we gebruiken de waardes die het boek ook aangeeft. Dus ik voorzie hier geen probleem mee (Zou eigenlijk wel minstens 0.4 moeten zijn ofzo)
    print(rejections/iters)
    w_samples_indices =  np.arange(10000, len(w_values), 1000)
    y_samples = [y(w_values[i]) for i in w_samples_indices]
    y_mean = np.mean(y_samples, axis = 0)
    #Waardes lijken met deze alpha ook te kloppen.
    print(y_mean)

    fig, axs = plt.subplots(4)

    axs[0].plot([i for i in range(iters)], [w[0] for w in w_values], label = "w0")
    axs[0].plot([i for i in range(iters)], [w[1] for w in w_values], label = "w1")
    axs[0].plot([i for i in range(iters)], [w[2] for w in w_values], label = "w2")
    axs[0].legend()

    axs[1].scatter([w[2] for w in w_values], [w[1] for w in w_values])
    axs[1].set_ylim([-3,5])
    axs[1].set_xlim([-1,7])

    axs[2].plot([i for i in range(iters)], [G(w) for w in w_values])
    axs[2].set_ylim([0,14])

    axs[3].plot([i for i in range(iters)], [M(w) for w in w_values])
    axs[3].set_ylim([0,14])

    fig.tight_layout(pad = 1.5)
    plt.show()


if __name__ == "__main__":
    main()


