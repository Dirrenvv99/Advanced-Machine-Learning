'''Compares time used between the two methods'''

from HMC import HMC
from MCMC import MH
import timeit
import numpy as np
from tqdm import tqdm
from functools import partialmethod
import matplotlib.pyplot as plt
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True) #silence tqdm
np.random.seed(0)

def measure_time(MCMC=True):
    means = []
    times = range(1,60,5)
    for time in times:
        start = timeit.default_timer()
        samples = []
        while timeit.default_timer()-start < time:
            if MCMC:
                sample, _ = MH(1,8)
            else: 
                sample, _ = HMC(30,0.03,1,int(np.ceil(8/(30*0.03)**2)))
                sample = sample[0]
            samples.append(sample)
        means.append(np.linalg.norm(np.mean(samples, axis=0)))
    return means
# measure_time(True)
def main():
    print(measure_time(False))
    print(measure_time(True))


def plot_means():
    #means found with the main function
    HMC_means = [0.10216311211424477, 0.032856734015629184, 0.036041019053063846, 0.07929965194003066, 0.001206175191324955, 0.03211537896670119, 0.017550417382544994, 0.0166879846992052, 0.004207068426866772, 0.013421548507685717, 0.00568575773927431, 0.010124926082536888]
    MH_means = [0.1800682580728149, 0.03714302851701778, 0.1721959963938036, 0.06837956401533761, 0.09672457618782034, 0.22230412141637193, 0.22240251165233163, 0.020521859420709714, 0.08336298331016413, 0.05430051397211834, 0.11466552730043512, 0.05690477394140244]
    plt.plot(range(1,60,5), HMC_means, label='HMC')
    plt.plot(range(1,60,5), MH_means, label='MH')
    plt.legend()
    plt.xlabel('time in seconds')
    plt.ylabel('norm of the mean')
    plt.title('HMC vs MH performace')
    plt.show()


if __name__ == '__main__':
    # main()
    plot_means()
