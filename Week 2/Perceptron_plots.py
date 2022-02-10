import numpy as np
sigmas = [0.001, 0.01, 0.1, 1] #numbers derived from perceptrion.py. They can be recreated by filling in the appropriate hyperparameters.
rejections = [0.08, 0.20, 0.52, 0.84]
burn_ins = [4337, 685, 188, 151]
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(rejections, burn_ins)
ax.set_xlabel('rejection rate')
ax.set_ylabel('burn-in time (in iterations)')
ax.set_title('rejection vs burn-in time for different proposal distributions')

for i, txt in enumerate(sigmas):
    ax.annotate('sigma = ' + str(txt), (rejections[i], burn_ins[i]))
plt.show()

epsilons = [0.005, 0.01, 0.05, 0.1]

rejections2 = [0.01, 0.02, 0.23, 0.68]

burn_ins2 = [0, 4, 45, 7]

fig, ax = plt.subplots()
ax.scatter(rejections2, burn_ins2)
ax.set_xlabel('rejection rate')
ax.set_ylabel('burn-in time (in iterations)')
ax.set_title('rejection vs burn-in time for different epsilon and tau values')

for i, txt in enumerate(epsilons):
    ax.annotate('epsilon = sqrt(' + str(txt)+')', (rejections2[i], burn_ins2[i]))
plt.show()