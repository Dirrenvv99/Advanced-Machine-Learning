import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description= 'Toy model BM')
parser.add_argument('-N',type= int, default= 20, help='Size of the dataset')
parser.add_argument('-S',type = int, default = 20, help = "Amount of spins" )
parser.add_argument('--eta',type = float, default = 0.005, help = "learningrate" )
parser.add_argument('--threshold',type = float, default = 10**(-13), help = "Threshold for convergence of the method" )
args = parser.parse_args()

lr = args.eta
threshold = args.threshold

def unnormalized_p(s, w,theta):
    return np.exp(0.5*np.dot(s,np.dot(w,s)) + np.dot(theta, s))

def p_s_w(s, data,w, theta):
    # for the exact model needs to be calculated exactly, thus including the normalization constant. 
    Z = np.sum([unnormalized_p(point,w,theta) for point in data])
    return 1/Z * unnormalized_p(s,w,theta)


def likelihood(data,w,theta):
    nom = 0 
    Z = np.sum([unnormalized_p(point,w,theta) for point in data])
    for s in data:
        nom += 0.5*(np.dot(s,np.dot(w,s))) + np.dot(theta, s)
    return nom/len(data) - np.log(Z)

def clamped_statistics(data):
    single = 1/(len(data)) * np.sum(data, axis = 0)
    data_needed = np.array([np.outer(x,x) for x in data])
    double = 1/(len(data)) * np.sum(data_needed, axis = 0)

    return single, double

def free_statistics(data, w, theta):
    single = np.sum(np.array([s * p_s_w(s,data,w,theta) for s in data]), axis = 0)
    double = np.sum(np.array([np.outer(s,s) * p_s_w(s,data,w,theta) for s in data]), axis = 0)
    return single, double


def main():
    #Create toy model dataset
    data = np.array([np.random.randint(0,2,size = args.S) for _ in range(args.N)])
    for index, point in enumerate(data):
        point[point==0] = -1
        data[index] = point

    #initialize w and theta randomly
    w = np.random.randn(args.S, args.S)
    np.fill_diagonal(w,0.) 
    theta = np.random.randn(args.S)

    condition = True
    likelihood_chain = [likelihood(data,w,theta)]
    i = 0
    while condition:
        single_clamped, double_clamped = clamped_statistics(data)
        single_free, double_free = free_statistics(data,w,theta)
        w += lr * (double_clamped - double_free)
        theta += lr * (single_clamped - single_free)
        likelihood_chain.append(likelihood(data,w,theta))

        if np.allclose(double_clamped, double_free, atol = threshold) and np.allclose(single_clamped, single_free, atol = threshold):
            condition = False
        
        i += 1
        # if i % 20 == 0:
        #     print("double: ", (double_clamped - double_free))
        #     print("single: ", (single_clamped - single_free))
        if i % 100 == 0:
            print("double: ",np.mean(np.abs(double_clamped - double_free)))
            print("single: ",np.mean(np.abs(single_clamped - single_free)))
            print(i)


    plt.plot([x for x in range(len(likelihood_chain))], likelihood_chain)
    plt.xlabel("iterations")
    plt.ylabel("likelihood")
    plt.title("Likelihood over iterations of exact BM for toy model")
    plt.show()

if __name__ == '__main__':
    main()
    






    




