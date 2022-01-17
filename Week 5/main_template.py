import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.sparse as sparse
from itertools import product

n=20
# Jth=0.1
full = True

def sprandsym(n, density):
    rvs = stats.norm().rvs
    X = sparse.random(n, n, density=density, data_rvs=rvs)
    upper_X = sparse.triu(X) 
    result = upper_X + upper_X.T - sparse.diags(X.diagonal())
    return result.toarray()


def mf_approx(n, m_ex, w, th, smoothing=.7, eps=10**-13):
    # Mean Field approximation
    m = np.random.normal(size=n) # random init

    dm = np.inf
    iterations = 0
    while(dm > eps):
        iterations += 1
        m_old = m
        m = smoothing * m + (1-smoothing) * np.tanh(np.dot(w,m) + th)
        dm = np.max(np.abs(m-m_old))
    
    error_mf = np.sqrt(1/n*np.sum(m-m_ex)**2) # final error

    return error_mf, iterations, m


def bp(n, m_ex, w, th, c, eps=10**-13):
    # Belief Propagation
    a = np.random.normal(size=(n,n))             # random init

    da = 1
    iterations = 0
    while(da > eps and iterations < 1000):
        iterations += 1
        a_old = a

        # print(th)
        # print(w)
        m_pos = 2*np.cosh(w + th + np.sum(np.multiply(a,c), axis=0))
        # print(m_pos)
        m_neg = 2*np.cosh(-w + th + np.sum(np.multiply(a,c), axis=0))
        # print(m_neg)

        a = .5 * (np.log(m_pos) - np.log(m_neg))
        # print(a)
        # exit()
        da = np.max(np.abs(a-a_old))

    m = np.tanh(np.sum(a, axis=0) + th)
    error_bp = np.sqrt(1/n*np.sum(m-m_ex)**2) # final error

    return error_bp, iterations, m
    

def main():
    np.random.seed(37) #TODO random over weights

    x = np.linspace(0, 2, num=10)
    error_mean, error_std = [], []
    iter_mean, iter_std = [], []
    chi_mean, chi_std = [], []

    for Jth in tqdm(x):
        errors, iters, error_chis = [],[],[]

        for _ in range(1):

            # toggle between full and sparse Ising network
            if full:                    # full weight matrix
                J0 = 0                  # J0 and J are as defined for the SK model
                J = 0.5
                w = J0/n+J/np.sqrt(n)*np.random.normal(size = (n,n))
                np.fill_diagonal(w,0)
                w = np.tril(w)+np.tril(w).T
                c = ~(w==0)             # neighborhood graph fully connected

            else:                       # sparse weight matrix
                c1 = 0.5                # connectivity is the approximate fraction of non-zero links in the random graph on n spins
                k = c1*n
                beta = 0.5
                w = sprandsym(n,c1)     # symmetric weight matrix w with c1*n^2 non-zero elements
                np.fill_diagonal(w,0)
                c = ~(w==0)             # sparse 0,1 neighborhood graph
                w = beta*((w>0).astype(int)-(w<0).astype(int)) # w is sparse with +/-beta on the links

            # Jth=.1
            # np.random.seed(0)
            th = np.random.normal(size = n)*Jth

            # EXACT
            sa = np.array(list(product([-1,1], repeat = n)))        # all 2^n spin configurations
            Ea = 0.5*np.sum(np.dot(sa,w)*sa,axis=1) + np.dot(sa,th) # array of the energies of all 2^n configurations
            Ea = np.exp(Ea)
            Z = np.sum(Ea)
            p_ex = Ea /Z             # probabilities of all 2^n configurations
            m_ex = np.dot(sa.T,p_ex) # exact mean values of n spins

            klad = np.outer(p_ex,np.ones(shape=(1,n)))*sa
            chi_ex = np.dot(sa.T,klad)-np.dot(m_ex,m_ex.T) # exact connected correlations

            # print(1)
            error_mf, iter_mf, m_mf = mf_approx(n,m_ex,w,th, smoothing=.5)
            # print(2)

            chi_mf = np.linalg.inv(np.eye(n)/(1-m_mf**2)-w)
            error_chi_mf = np.sqrt(2/(n*(n-1))*np.sum(np.tril(chi_mf - chi_ex, -1)**2))
            # print(np.sum(np.tril(chi_ex-chi_mf, -1)))
            # error_chi_mf = np.sqrt(1/(n**2)*np.sum(chi_ex-chi_mf)**2)
            # print(error_chi_mf)
            # print() # np.sqrt(np.mean(np.square(np.dot(sa.T,klad)-np.dot(m_ex,m_ex.T))))
            # print("MF APPROX")
            # print(f"ERROR: {error_mf},\tITER: {iter_mf},\tCHI: {chi_mf}\n")

            error_bp, iter_bp, m_bp = bp(n,m_ex,w,th, c)
            # print(3)

            error_chi_bp = 0#np.dot(sa.T,klad)-np.dot(m_bp,m_bp.T) # exact connected correlations
            # print("BP")
            # print(f"ERROR: {error_bp},\tITER: {iter_bp},\tCHI: {chi_bp}\n")

            errors.append([error_mf, error_bp])
            iters.append([iter_mf, iter_bp])
            error_chis.append([error_chi_mf, error_chi_bp])

        error_mean.append(np.mean(errors, axis=0))
        error_std.append(np.std(errors, axis=0))
        iter_mean.append(np.mean(iters, axis=0))
        iter_std.append(np.std(iters, axis=0))
        chi_mean.append(np.mean(error_chis, axis=0))
        chi_std.append(np.std(error_chis, axis=0))

    def helper(plot, x, mean, std):
        mf_mean = np.array([i[0] for i in mean])
        mf_std = np.array([i[0] for i in std])
        plot.plot(x, mf_mean, label="mf")
        plot.fill_between(x, mf_mean-mf_std, mf_mean+mf_std, alpha=.5)

        bp_mean = np.array([i[1] for i in mean])
        bp_std = np.array([i[1] for i in std])
        plot.plot(x, bp_mean, label="bp")
        plot.fill_between(x, bp_mean-bp_std, bp_mean+bp_std, alpha=.5)

        plot.set_xlabel(r'$\beta$')
        plot.legend()

    _, axs = plt.subplots(1,3)

    axs[0].set_title('error')
    helper(axs[0], x, error_mean, error_std)
    axs[0].set_xlabel(r'$\beta$')
    axs[0].legend()

    axs[1].set_title('iterations')
    helper(axs[1], x, iter_mean, iter_std)
    axs[1].set_xlabel(r'$\beta$')
    axs[1].legend()

    axs[2].set_title('error chi')
    helper(axs[2], x, chi_mean, chi_std)
    axs[2].set_xlabel(r'$\beta$')
    axs[2].legend()

    plt.show()


if __name__ == '__main__':
    main()
