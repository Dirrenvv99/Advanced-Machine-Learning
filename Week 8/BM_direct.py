import numpy as np
import matplotlib.pyplot as plt
import argparse
from itertools import product
from collections import Counter


parser = argparse.ArgumentParser(description='Toy model BM')
parser.add_argument('-N',type=int, default=1000, help='Size of the dataset')
parser.add_argument('-S',type=int, default=10, help="Amount of spins")
args = parser.parse_args()


def unnormalized_p(s, w,theta):
    return np.exp(0.5*np.dot(s,np.dot(w,s)) + np.dot(theta, s))


def p_s_w(s, w, theta, Z):
    # for the exact model needs to be calculated exactly, thus including the normalization constant.
    return 1/Z * unnormalized_p(s, w, theta)


# def likelihood(w,theta, all_states, data):
#     Z = np.sum(unnormalized_p(all_states, w, theta))
#     nom = [0.5*np.dot(s,np.dot(w, s)) + np.dot(theta, s) for s in data] 
#     return np.mean(nom, axis=0) - np.log(Z)


def clamped_statistics(data):
    single = 1/(len(data)) * np.sum(data, axis=0)
    data_needed = np.array([np.outer(x, x) for x in data])
    double = 1/(len(data)) * np.sum(data_needed, axis=0)
    # Diagonals are set to zero. 
    # Since the diagonal of the w should be zero anyway.
    np.fill_diagonal(double, 0.)
    return single, double


if __name__ == '__main__':
    full_data = np.loadtxt("bint.txt")
    data = full_data[:,:953]
    data = data[np.random.choice(range(160), size=10, replace=False)]
    data = data.transpose()
    print("data retreived")
    # seed to make sure it can be recreated
    # np.random.seed(42)
    # indices_10 = np.random.choice(range(160), size = args.N, replace = False)
    # data = data_before[indices_10]

    # Create toy model dataset
    # data = np.array([np.random.randint(0, 2, size=args.S) for _ in range(args.N)])

    clamped_single, clamped_double = clamped_statistics(data)
    C = clamped_double - np.outer(clamped_single, clamped_single)
    C = C + np.eye(*C.shape)*10**-13
    m = clamped_single

    w = np.zeros_like(C)
    np.fill_diagonal(w, 1/(1-m**2))
    w = w - np.linalg.inv(C)

    theta = np.arctanh(m) - np.dot(w, m)

    print(w.shape, theta.shape)

    full_data = data.tolist()
    new_data = map(tuple, full_data)
    new_data_set = [list(item) for item in set(tuple(row) for row in full_data)]

    print(new_data_set[0])

    observed_occ = Counter(new_data)

    print(observed_occ)

    observed_rate = []
    for i in new_data_set:
        observed_rate.append(observed_occ[tuple(i)]/len(full_data))

    all_states = list(product(range(2), repeat = len(data[0])))

    Z = np.sum([unnormalized_p(np.array(s), w, theta) for s in all_states])

    approximated_rate = [p_s_w(np.array(s), w, theta, Z) for s in new_data_set]
    print(len(observed_rate), len(approximated_rate), observed_rate[0], approximated_rate[0])

    plt.scatter(observed_rate, approximated_rate, label="patterns_dep", color="red")
    plt.plot([x for x in np.linspace(0.000001,1, 10000)],[x for x in np.linspace(0.000001,1, 10000)], color = "red", label = "y = x")
    plt.xlabel("Observerd pattern rate")
    plt.ylabel("Approximated by BM pattern rate")
    plt.xscale('log')
    plt.legend()
    plt.yscale('log')
    plt.title("Recreation fig 2a")
    plt.show()
