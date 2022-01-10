from HMC import HMC
from MCMC import MH
import timeit
import numpy as np
np.random.seed(0)

def measure_time(MCMC=True):
    start = timeit.default_timer()
    means = []
    times = range(1,30,5)
    for time in times:
        samples = []
        while timeit.default_timer()-start < time:
            if MCMC:
                sample, _ = MH(1,8)
            else: 
                sample, _ = HMC(30,0.03,1,int(np.ceil(8/(30*0.03)**2)))
                sample = sample[0]
            samples.append(sample)
        means.append(np.mean(samples, axis=0))
    print(means)
 
measure_time(True)
measure_time(False)
