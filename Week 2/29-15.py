import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
MK exercise 29.15.
Gibbs sampling of posterior over mu, sigma given data (5).
Hint: It is recommended to sample beta = 1/sigma^2 rather than sigma^2.
But be aware that such a transformation affects the prior that you assume.
For instance, if you assume a flat prior over sigma^2, this transforms to a non-flat prior over beta.
For this exercise choose the prior over beta as 1/beta.
This choice corresponds to a so-called non-informative prior that is flat in the log(sigma) domain.
See also slides lecture 3 where we consider the variational approximation for this problem. (5) 
"""

def gaus(x, mu, sigma):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2*sigma**2))


def gaus_sample(mu, sigma):
    return np.random.normal(mu, sigma)


def gamma(x, shape, scale):
    return x**(shape-1)*(np.exp(-x/scale) / (sps.gamma(shape)*scale**shape))


def gaus_gamma(shape, scale):
    return np.random.gamma(shape, scale, 1000)


def main():
    mu = 0
    sigma = 5
    x = np.arange(-20, 20, .01)
    y = gaus(x, mu, sigma)

    fig, ax = plt.subplots()
    ax.plot(x, y)

    # def p(x, sigma_p):
    #     return scipy.stats.norm.pdf(x, loc = 0, scale = sigma_p)

    # def q(x, sigma_q):
    #     return scipy.stats.norm.pdf(x, loc = 0, scale = sigma_q)

    # def sample_p(sigma_p):
    #     return scipy.stats.norm.rvs(loc = 0, scale = sigma_p)

    # def sample_q(sigma_q):
    #     return scipy.stats.norm.rvs(loc = 0, scale = sigma_q)

    # def normalizing_constant(N, sigma_q, sigma_p = 1):
    #     np.random.seed(0)
    #     samples = [sample_q(sigma_q) for _ in range(N)]
    #     estimated_z = np.mean([p(x,sigma_p)/q(x,sigma_q) for x in samples])
    #     return estimated_z

    # def emperical_std(N,sigma_q, sigma_p = 1):
    #     np.random.seed(0)
    #     samples = [sample_q(sigma_q) for _ in range(N)]
    #     emp_std = np.std([p(x,sigma_p)/q(x,sigma_q) for x in samples])
    #     return emp_std

    # def weights(W, sigma_p = 1):         

    #     xs = [sample_q(1) for _ in range(W)]
    #     for x in xs:
    #         axs[2].plot([sigma for sigma in np.linspace(0.1,1.6,100)], [p(x,sigma)/q(x,1) for sigma in np.linspace(0.1,1.6,100)])
    #         axs[2].set_ylim([-.1,3.5])

    # for N in tqdm(N_values):
    #     axs[0].plot([sigma for sigma in np.linspace(0.02,1.6,25)], [normalizing_constant(N, sigma) for sigma in np.linspace(0.02,1.6,25)], label = str(N))
    #     axs[0].legend()   
    
    # for N in tqdm(N_values):
    #     axs[1].plot([sigma for sigma in np.linspace(0.02,1.6,25)], [emperical_std(N, sigma) for sigma in np.linspace(0.02,1.6,25)], label = str(N))
    #     axs[1].legend()

    # weights(30)
    
    plt.show()


if __name__ == '__main__':
    main()