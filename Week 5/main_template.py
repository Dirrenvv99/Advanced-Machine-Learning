from cProfile import label
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.sparse as sparse
from itertools import product

n=20
full = False

def sprandsym(n, density):
    rvs = stats.norm().rvs
    X = sparse.random(n, n, density=density, data_rvs=rvs)
    upper_X = sparse.triu(X) 
    result = upper_X + upper_X.T - sparse.diags(X.diagonal())
    return result.toarray()


def mf_approx(n, m_ex, w, th, smoothing=0, eps=10**-13):
    # Mean Field approximation
    m = np.random.normal(size=n) # random init
    
    dm = np.inf
    iterations = 0
    while(dm > eps):
        iterations += 1
        if (iterations % 1000 == 0):
            print("MF: ", iterations)
        m_old = m
        m = smoothing*m + (1-smoothing)*np.tanh(np.dot(w,m) + th)
        dm = np.max(np.abs(m-m_old))

    error_mf = np.sqrt(1/n*np.sum((m-m_ex)**2)) # final error

    # plt.figure()
    # plt.ylabel('iterations')
    # plt.xlabel(r'$\eta$')
    # plt.plot(np.linspace(0, 1, num=20), iters)
    # plt.show()
    # exit()

    # plt.figure()
    # plt.ylabel('dm')
    # plt.xlabel('iterations')
    # plt.yscale('log')
    # plt.plot(dms2, label="with smoothing")
    # plt.plot(dms1, label="without smoothing")
    # plt.legend()
    # plt.show()
    # exit()

    return error_mf, iterations, m


def bp(n, m_ex, w, th, c, eps=10**-13, smoothing=0):
    # Belief Propagation
    a = np.random.normal(size=(n,n))             # random init
    da = 1
    iterations = 0
    while(da > eps and iterations < 1500):
        iterations += 1
        if (iterations % 1000 == 0):
            print("BP: ", iterations)
        a_old = a

        m_pos = 2*np.cosh(w + th + np.sum(np.multiply(a,c), axis=1) - np.multiply(a,c).T)
        m_neg = 2*np.cosh(-w + th + np.sum(np.multiply(a,c), axis=1) - np.multiply(a,c).T)

        a = .5 * (np.log(m_pos) - np.log(m_neg))
        a = smoothing*a_old + (1-smoothing)*a
        da = np.max(np.abs(a-a_old))

    m = np.tanh(np.sum(a, axis=1) + th)
    error_bp = np.sqrt(1/n*np.sum((m-m_ex)**2)) # final error

    # plt.figure()
    # plt.ylabel('iterations')
    # plt.xlabel(r'$\eta$')
    # plt.plot(np.linspace(0, 1, num=20), iters)
    # plt.show()
    # exit()

    # plt.figure()
    # plt.ylabel('da')
    # plt.xlabel('iterations')
    # plt.yscale('log')
    # plt.plot(das2, label="with smoothing")
    # plt.plot(das1, label="without smoothing")
    # plt.legend()
    # plt.show()
    # exit()

    return error_bp, iterations, m, a

def bij_func(i,j,xi,xj,w,th,a,c):
    part1 = w[i][j]*xi*xj
    part2 = th[i]*xi + th[j]*xj
    part3 = np.sum([a[k][i]*xi*c[k][i] for k in range(n)]) - a[j][i]*xi*c[j][i]
    part4 = np.sum([a[l][j]*xj*c[l][j] for l in range(n)]) - a[i][j]*xj*c[i][j]
    return np.exp(part1 + part2 + part3 + part4)

def b_ij(i,j,xi,xj,w,th,a,c,Z=0):
    if Z==0:
        for xi in [-1,1]:
            for xj in [-1,1]:
                Z += bij_func(i,j,xi,xj,w,th,a,c)
        return Z
    else:
        return bij_func(i,j,xi,xj,w,th,a,c)/Z


def X_ij(i,j,w,th,a,c,m):
    total = -m[i]*m[j]
    Z = b_ij(i,j,0,0,w,th,a,c)
    for xi in [-1,1]:
        for xj in [-1,1]:
            total +=  b_ij(i,j,xi,xj,w,th,a,c,Z)*xi*xj
    return total

def main():
    np.random.seed(37)

    x = np.linspace(0.1, 1, num=9)
    _, axs = plt.subplots(1,3) if full else plt.subplots(1,2)

    # for Jth in tqdm(x):
    Jth = .1
    for c1 in [0.1, 0.6, 1]:
        error_mean, error_std = [], []
        iter_mean, iter_std = [], []
        chi_mean, chi_std = [], []
        for beta in tqdm(x):
            errors, iters, error_chis = [],[],[]

            for _ in tqdm(range(6)):

                # toggle between full and sparse Ising network
                if full:                    # full weight matrix
                    J0 = 0                  # J0 and J are as defined for the SK model
                    J = Jth                 # previously 0.5
                    w = J0/n+J/np.sqrt(n)*np.random.normal(size = (n,n))
                    np.fill_diagonal(w,0)
                    w = np.tril(w)+np.tril(w).T
                    c = ~(w==0)             # neighborhood graph fully connected
                    th = np.random.normal(size = n)*Jth

                else:                       # sparse weight matrix
                    # c1 = 0.5                # connectivity is the approximate fraction of non-zero links in the random graph on n spins
                    k = c1*n
                    # beta = 0.2
                    w = sprandsym(n,c1)     # symmetric weight matrix w with c1*n^2 non-zero elements
                    np.fill_diagonal(w,0)
                    c = ~(w==0)             # sparse 0,1 neighborhood graph
                    w = beta*((w>0).astype(int)-(w<0).astype(int)) # w is sparse with +/-beta on the links
                    th = np.full(n, 1)


                # EXACT
                sa = np.array(list(product([-1,1], repeat = n)))        # all 2^n spin configurations
                Ea = 0.5*np.sum(np.dot(sa,w)*sa,axis=1) + np.dot(sa,th) # array of the energies of all 2^n configurations
                Ea = np.exp(Ea)
                Z = np.sum(Ea)
                p_ex = Ea /Z             # probabilities of all 2^n configurations
                m_ex = np.dot(sa.T,p_ex) # exact mean values of n spins

                klad = np.outer(p_ex,np.ones(shape=(1,n)))*sa
                chi_ex = np.dot(sa.T,klad)-np.outer(m_ex,m_ex.T) # exact connected correlations

                error_mf, iter_mf, m_mf = mf_approx(n,m_ex,w,th, smoothing=0.5)

                chi_mf = np.linalg.inv(np.eye(n)/(1-m_mf**2)-w)
                error_chi_mf = np.sqrt(2/(n*(n-1))*np.sum(np.tril(chi_mf - chi_ex, -1)**2))

                error_bp, iter_bp, m_bp, a= bp(n,m_ex,w,th, c, smoothing=0.5)
                chi_bp = np.empty(shape=(n,n))
                for i in range(n):
                    for j in range(n):
                        chi_bp[i][j] = X_ij(i,j,w,th,a,c,m_bp)

                error_chi_bp = np.sqrt(2/(n*(n-1))*np.sum(np.tril(chi_bp - chi_ex, -1)**2)) # exact connected correlations

                errors.append([error_mf, error_bp])
                iters.append([iter_mf, iter_bp])
                error_chis.append([error_chi_mf, error_chi_bp])

            error_mean.append(np.mean(errors, axis=0))
            error_std.append(np.std(errors, axis=0))
            iter_mean.append(np.mean(iters, axis=0))
            iter_std.append(np.std(iters, axis=0))
            chi_mean.append(np.mean(error_chis, axis=0))
            chi_std.append(np.std(error_chis, axis=0))

        def helper(plot, x, mean, std, label=""):
            mf_mean = np.array([i[0] for i in mean])
            mf_std = np.array([i[0] for i in std])
            plot.plot(x, mf_mean, label="mf"+label)
            plot.fill_between(x, mf_mean-mf_std, mf_mean+mf_std, alpha=.3)

            bp_mean = np.array([i[1] for i in mean])
            bp_std = np.array([i[1] for i in std])
            plot.plot(x, bp_mean, label="bp"+label)
            plot.fill_between(x, bp_mean-bp_std, bp_mean+bp_std, alpha=.3)

        helper(axs[0], x, error_mean, error_std, f" c={c1}")
        helper(axs[1], x, iter_mean, iter_std, f" c={c1}")
        if full:
            helper(axs[2], x, chi_mean, chi_std)

    var = r'$\beta$'
    axs[0].set_title('error')
    axs[0].set_xlabel(var)
    if not full:
        axs[0].set_yscale('log')
    axs[0].legend()
    axs[1].set_title('iterations')
    axs[1].set_xlabel(var)
    if not full:
        axs[1].set_ylim(0, 2000)
    axs[1].legend()
    if full:
        axs[2].set_title('error chi')
        axs[2].set_xlabel(var)
        axs[2].legend()

    plt.show()


if __name__ == '__main__':
    main()
