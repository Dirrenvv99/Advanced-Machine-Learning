import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
A = 250.25
B = -249.75

def E(x):
    # return 0.5*(250.25*x[0]**2 - x[0]*x[1]*2*249.75 + 250.25*x[1])
    return 0.5*A*(x[0]**2+x[1]**2) + B*x[0]*x[1]


def gradE(x):
    return np.array([A* x[0] + B*x[1], A* x[1] + B*x[0]])


def HMC(tau, eps, nr_samples, iter = 100):
    samples = []
    #amples = np.random.multivariate_normal(np.array([0,0]), np.eye(2), 1)[0]
    rejections = []
    for _ in tqdm(range(nr_samples)):
        rejection_count = 0
        x = np.random.multivariate_normal(np.array([0,0]), np.eye(2), 1)[0]
        #g = gradE(x)
        for _ in range(iter):                
            p = np.random.multivariate_normal(np.array([0,0]), np.eye(2), 1)[0]
            
            H_old = np.dot(p,p)/2 + E(x)
            x_old = np.copy(x)

            for _ in range(tau):    # Leapfrog
                p -= eps * gradE(x)/2
                x += eps * p
                p -= eps * gradE(x)/2
            
            H_new = np.dot(p,p)/2 + E(x)
            dH = H_new - H_old

            if dH != dH:
                print("something went wrong and produces NaN values")

            if dH < 0 or np.random.random() < np.exp(-dH):
                x = x    #just to make this work, and for clarity; it shows that after the steps x has become x_new, and if x_new is rejected, we just revert x back to x_old               
            else:
                x = x_old
                rejection_count += 1
        samples.append(x)
        rejections.append(rejection_count)
    return samples, rejections


def main():
    nr_samples = 100
    iters = 1000
    taus = [4]
    epss = [0.0155]
    fig, axs = plt.subplots(len(epss), len(taus))

    real_samples = np.random.multivariate_normal([0,0], [[A,B],[B,A]], 100)

    samples, rejections = HMC(taus[0], epss[0], nr_samples, iters)
    
    print(f"{taus[0]},\t{epss[0]}\t{np.mean([rejection/iters for rejection in rejections])}")
   
    # print(samples)

    plt.scatter([x[0] for x in real_samples], [x[1] for x in real_samples], color = "blue", label = "normally sampled", marker = ".")
    plt.scatter([x[0] for x in samples], [x[1] for x in samples], color = "red", label = "samples")
    plt.title(f"sample plot with tau: {taus[0]}; epsilon: {epss[0]}")
    plt.legend()

    # for tau, eps in tqdm([(i, j) for i in taus for j in epss]):
    #     samples, rejections = HMC(tau, eps, nr_samples, iters)
        
    #     print(f"{tau},\t{eps}\t{np.mean([rejection/iters for rejection in rejections])}")
    #     ei, ti = epss.index(eps), taus.index(tau)
    #     # print(samples)

    #     axs[ei, ti].scatter([x[0] for x in real_samples], [x[1] for x in real_samples], color = "blue", label = "normally sampled", marker = ".")
    #     axs[ei, ti].scatter([x[0] for x in samples], [x[1] for x in samples], color = "red", label = "samples")
    #     axs[ei, ti].set_title(f"sample plot with tau: {tau}; epsilon: {eps}")
    #     if (ei+ti==0):
    #         axs[ei, ti].legend()
    # fig.tight_layout(pad = 3.0)
    plt.show()
        

if __name__ == '__main__':
    main()


"""
Optimal value Approx:
Sigma          ~
Rejection Rate ~
"""