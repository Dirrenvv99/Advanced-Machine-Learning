import numpy as np
import scipy.special as sps
from scipy.stats import multivariate_normal
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


def gamma_sample(shape, scale):
    return np.random.gamma(shape, scale)


def update_step(beta, mu_init_variance, data_sum, data, N):
    var_sample  = 1/(1/mu_init_variance + beta*N)
    mu_sample = data_sum/(1/(beta*mu_init_variance) + N)
 
    mu_next = gaus_sample(mu_sample, np.sqrt(var_sample))

    gamma_shape = N/2
    gamma_scale_denom = np.sum(np.array([(x - mu_next)**2 for x in data]))
    gamma_scale = 2/gamma_scale_denom

    beta_next = gamma_sample(gamma_shape, gamma_scale)

    return mu_next, beta_next
    



def gibbs_sampling(mu_init, beta_init, data, N, mu_init_variance, iter = 100):
    data_sum = np.sum(data)
    mu_prev = mu_init
    beta_prev = beta_init
    for _ in range(iter - 1):
        mu_next , beta_next = update_step(beta_prev, mu_init_variance, data_sum, data, N)
        mu_prev, beta_prev = mu_next, beta_next
    
    mu_final, beta_final = update_step(beta_next, mu_init_variance, data_sum, data, N)
    return mu_final , beta_final




def main():
    # mu = 0
    # sigma = 5
    # x = np.arange(-20, 20, .01)
    # y = gaus(x, mu, sigma)

    N = 1000 # Number of data points
    real_mu = 3
    real_scale = 42 # note that this is the scale -> np.sqrt(sigma) = 42 -> sigma = 42^2 = 1764-> precision (beta) = 1/1764

    data = np.array([gaus_sample(real_mu, real_scale) for _ in range(N)])
    
    #start with two initial mu and beta:
    mu_init_variance = 50
    mu_init = gaus_sample(0,np.sqrt(mu_init_variance))
    beta_init = 1/2 #just a random choice

    mus = []
    betas = [] 
    for _ in tqdm(range(500)):
        mu_final, beta_final = gibbs_sampling(mu_init, beta_init, data, N, mu_init_variance, iter = 100)
        mus.append(mu_final)
        betas.append(beta_final)

    print("beta mean found:" , np.mean(betas), "real is: ", 1/real_scale**2)
    print("mu mean found: ", np.mean(mus), "real is: ", real_mu, "with an std of: ", np.std(mus)) 


if __name__ == '__main__':
    main()