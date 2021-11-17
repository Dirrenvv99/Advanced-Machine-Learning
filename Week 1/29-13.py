import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from tqdm import tqdm


def main():

    N_values = [1000,10000]

    fig, axs = plt.subplots(1,3)

    def p(x, sigma_p):
        return scipy.stats.norm.pdf(x, loc = 0, scale = sigma_p)

    def q(x, sigma_q):
        return scipy.stats.norm.pdf(x, loc = 0, scale = sigma_q)

    def sample_p(sigma_p):
        return scipy.stats.norm.rvs(loc = 0, scale = sigma_p)

    def sample_q(sigma_q):
        return scipy.stats.norm.rvs(loc = 0, scale = sigma_q)

    def normalizing_constant(N, sigma_q, sigma_p = 1):
        samples = [sample_q(sigma_q) for _ in range(N)]
        estimated_z = np.mean([p(x,sigma_p)/q(x,sigma_q) for x in samples])
        return estimated_z

    def emperical_std(N,sigma_q, sigma_p = 1):
        samples = [sample_q(sigma_q) for _ in range(N)]
        emp_std = np.std([p(x,sigma_p)/q(x,sigma_q) for x in samples])
        return emp_std

    def weights(W, sigma_p = 1):         

        xs = [sample_q(1) for _ in range(W)]
        for x in xs:
            axs[2].plot([sigma for sigma in np.linspace(0.1,1.6,100)], [p(x,sigma_p)/q(x,sigma) for sigma in np.linspace(0.1,1.6,100)])

    # for N in tqdm(N_values):
    #     axs[0].plot([sigma for sigma in np.linspace(0.02,1.6,25)], [normalizing_constant(N, sigma) for sigma in np.linspace(0.02,1.6,25)], label = str(N))
    #     axs[0].legend()   
    
    # for N in tqdm(N_values):
    #     axs[1].plot([sigma for sigma in np.linspace(0.02,1.6,25)], [emperical_std(N, sigma) for sigma in np.linspace(0.02,1.6,25)], label = str(N))
    #     axs[1].legend()

    weights(30)
    

    plt.show()




if __name__ == '__main__':
    main()