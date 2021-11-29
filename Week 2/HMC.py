import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def E(x):
    # return 0.5*(250.25*x[0]**2 - x[0]*x[1]*2*249.75 + 250.25*x[1])
    return 0.5*250.25*(x[0]**2+x[1]**2) + 249.75*x[0]*x[1]


def gradE(x):
    return np.array([250.25* x[0] - 249.75*x[1], 250.25* x[1] - 249.75*x[0]])


def HMC(tau, eps, epochs):
    np.random.seed(0)
    samples = [np.random.multivariate_normal(np.array([0,0]), np.eye(2), 1)[0]]
    rejections = 0
    for _ in tqdm(range(epochs)):
        x = np.copy(samples[-1])
        g = gradE(x)
        p = np.random.multivariate_normal(np.array([0,0]), np.eye(2), 1)[0]
        
        H_old = np.dot(p,p)/2 + E(x)
        x_new = np.copy(x)

        for _ in range(tau):    # Leapfrog
            p -= eps * g/2
            x_new += eps * p
            g = gradE(x_new)
            p -= eps * g/2
        
        H_new = np.dot(p,p)/2 + E(x_new)
        dH = H_new - H_old

        if dH != dH:
            print("something went wrong and produces NaN values")

        if dH < 0 or np.random.random() < np.exp(-dH):
            samples.append(x_new)

        else:
            # samples.append(x)
            rejections += 1

    return samples, rejections


def main():
    epochs = 10000

    taus = [5, 10]
    epss = [i*0.001 for i in range(10, 100, 30)] #.05, .1]
    fig, axs = plt.subplots(len(epss), len(taus))

    np.random.seed(0)
    real_samples = np.random.multivariate_normal([0,0], [[250.25,-249.75],[-249.75,250.25]], 100)

    for tau, eps in [(i, j) for i in taus for j in epss]:
        samples, rejections = HMC(tau, eps, epochs)
        
        print(f"{tau},\t{eps}\t{rejections}")
        ei, ti = epss.index(eps), taus.index(tau)
        # print(samples)

        axs[ei, ti].scatter([x[0] for x in real_samples], [x[1] for x in real_samples], color = "blue", label = "normally sampled", marker = ".")
        axs[ei, ti].scatter([x[0] for x in samples], [x[1] for x in samples], color = "red", label = "samples")
        axs[ei, ti].set_title(f"sample plot with tau: {tau}; epsilon: {eps}")
        if (ei+ti==0):
            axs[ei, ti].legend()
    # fig.tight_layout(pad = 3.0)
    plt.show()
        

if __name__ == '__main__':
    main()


"""
Optimal value Approx:
Sigma          ~
Rejection Rate ~
"""