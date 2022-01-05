import numpy as np
# from scipy.stats import multivariate_normal
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.sparse as sparse
from itertools import product

np.random.seed(37)
n=3
Jth=0.1
full = False

def sprandsym(n, density):
    rvs = stats.norm().rvs
    X = sparse.random(n, n, density=density, data_rvs=rvs)
    upper_X = sparse.triu(X) 
    result = upper_X + upper_X.T - sparse.diags(X.diagonal())
    return result.toarray()
def main():
    if full: # toggle between full and sparse Ising network
        # full weight matrix   
        J0=0                        # J0 and J are as defined for the SK model
        J=0.5
        w=J0/n+J/np.sqrt(n)*np.random.normal(size = (n,n))
        np.fill_diagonal(w,0)
        w=np.tril(w)+np.tril(w).T
        c =~(w==0)                 # neighborhood graph fully connected
    else:
        # sparse weight matrix
        c1=0.5                         #connectivity is the approximate fraction of non-zero links in the random graph on n spins
        k=c1*n
        beta=0.5
        w=sprandsym(n,c1)          #symmetric weight matrix w with c1*n^2 non-zero elements
        np.fill_diagonal(w,0)
        c =~(w==0)                  # sparse 0,1 neighborhood graph 
        w=beta*((w>0).astype(int)-(w<0).astype(int))             # w is sparse with +/-beta on the links
    th = np.random.normal(size = n)*Jth
    # w *= 10
    #EXACT
    sa = np.array(list(product([-1,1], repeat = n)))         #all 2^n spin configurations
    Ea = 0.5*np.sum(np.dot(sa,w)*sa,axis=1) + np.dot(sa,th) # array of the energies of all 2^n configurations
    Ea=np.exp(Ea)
    Z=np.sum(Ea)
    p_ex = Ea /Z  # probabilities of all 2^n configurations
    m_ex= np.dot(sa.T,p_ex)  # exact mean values of n spins
    klad=np.outer(p_ex,np.ones(shape=(1,n)))*sa
    # print(klad)
    # print(m_ex)
    # print(np.sum(klad, axis=0))
    chi_ex=np.dot(sa.T,klad)-np.dot(m_ex,m_ex.T) # exact connected correlations
    # print(chi_ex)

    # MF
    # write your code
    smoothing = 0
    m = np.random.normal(size=n) #random init
    # print(m)
    error_mf = np.sqrt(1/n*np.sum(m-m_ex)**2)
    # print(error_mf)

    eps = 10**-13
    dm = np.inf
    temp = 0
    while(dm > eps):
        # print(m)
        temp += 1
        m_old = m
        m = smoothing*m +(1-smoothing)*np.tanh(np.dot(w,m) + th)
        dm = np.max(np.abs(m-m_old))
    
    error_mf = np.sqrt(1/n*np.sum(m-m_ex)**2)
    print(m)
    print(m_ex)
    # print(error_mf)
    
    # print(temp)
    # print(w)
    # print(c)

    # %BP
    # %write your code
    # error_bp=sqrt(1/n*sum(m_bp-m_ex).^2)
    a = np.random.normal(size=(n,n))
    da = 1
    temp = 0
    while(da > eps):
        temp += 1
        a_old = a
        
        m_pos = 2*np.cosh(w + th + np.sum(a, axis=1))   # axis =1 misschien niet goed
        m_neg = 2*np.cosh(-w + th + np.sum(a, axis=1))

        a = .5 * (np.log(m_pos) - np.log(m_neg))

        da = np.max(np.abs(a-a_old))
    m_bp = np.tanh(th + np.sum(a, axis=1))
    print(m_bp)

if __name__ == '__main__':
    main()