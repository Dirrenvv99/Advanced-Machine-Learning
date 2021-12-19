import numpy as np
# import scipy.special as sps
# from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

"""
Consider again the perceptron learning problem of Mackay chapter 39 and 41,
for which we computed the posterior by sampling in week 2.
This time, compute p(t=1|x,D,alpha) using the Laplace approximation and
reproduce Mackay figure 41.11b.
"""

t = np.loadtxt("t.txt")
x = np.loadtxt("x.txt")
# t = np.loadtxt("t.txt")[10:]
# x = np.loadtxt("x.txt")[10:]
# t_test = np.loadtxt("t.txt")[-1:]
# x_test = np.loadtxt("x.txt")[-1:]


def f(w):
    return np.dot(x, w)


def sigmoid(y):
    return 1/(1+np.exp(-y))


def G(w):
    e = 10**(-77)
    y_list = sigmoid(f(w))

    result = 0
    for idx, y in enumerate(y_list):
        if y == 1:
            result -= t[idx]*np.log(y) + (1-t[idx])*np.log(1-y-e)
        elif y == 0:
            result -= t[idx]*np.log(y+e) + (1-t[idx])*np.log(1-y)
        else:
            result -= t[idx]*np.log(y) + (1-t[idx])*np.log(1-y)

    return result


def M(w, alpha):
    return G(w) + .5*alpha*np.dot(w,w)


def grad_G(w):
    result = np.array([.0,.0,.0])
    y_list = f(w)
    for i in range(len(y_list)):
        result -= t[i] * (1 - y_list[i]) * x[i] - (1-t[i]) * y_list[i] * x[i]
    return result 


def grad_E(w):
    return w


def grad_M(w, alpha):
    # epsilon = 10**-5
    return grad_G(w) + alpha * grad_E(w)
    # return np.array([
    #     (M(w, alpha)+M(w+epsilon*np.array([1,0,0]), alpha))/epsilon,
    #     (M(w, alpha)+M(w+epsilon*np.array([0,1,0]), alpha))/epsilon,
    #     (M(w, alpha)+M(w+epsilon*np.array([0,0,1]), alpha))/epsilon])


def hessian(w, alpha):
    _, d = x.shape
    h = f(w)

    hess = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            hess[i,j] = np.sum(sigmoid(h) * sigmoid(-h) * x[:,i] * x[:,j])

    return hess + alpha*np.identity(d)


def gradient_descent(w, alpha, learning_rate, iterations, grad_min):
    result = [np.copy(w)]

    if iterations:
        for _ in range(iterations):
            grad = grad_M(w, alpha)
            if (grad < grad_min).all():
                break
            w -= learning_rate * grad
            result.append(np.copy(w))
    else:
        grad = np.ones_like(w) * np.inf
        while not (np.abs(grad) < grad_min).all():
            # print(np.abs(grad)[0])
            # print((np.abs(grad)[0] < grad_min))
            grad = grad_M(w, alpha)
            print(grad)
            if (grad != grad).any() or (grad == np.inf).any():
                break
            w -= learning_rate * grad
            result.append(np.copy(w))

    return result


def l_approx(w, alpha, grad_lr, grad_iter, grad_min):
    w_values = gradient_descent(w, alpha, grad_lr, grad_iter, grad_min)
    hinv_values = []

    for w_star in w_values:
        h = hessian(w_star, alpha)
        hinv = np.linalg.inv(h)
        hinv_values.append(hinv)

    return w_values, hinv_values


def main():
    np.random.seed(0)
    alpha = 0.5
    grad_lr, grad_iter, grad_min = .005, None, 10**-10
    # w = np.random.multivariate_normal([0,0,0], np.eye(3))
    w = np.array([3.,3.,3.])

    w_values, hinv_values = l_approx(w, alpha, grad_lr, grad_iter, grad_min)

    scores = []

    w_star = w_values[-1]
    for w_star, hinv in zip(w_values, hinv_values):
        approx = np.array([])
        for sample in x:
            a_star = np.dot(sample, w_star)
            s_squared = np.dot(np.dot(sample, hinv), sample)

            kappa = 1/np.sqrt(1+np.pi*s_squared/8)
            psi = sigmoid(kappa*a_star)
            approx = np.append(approx, psi)

        scores.append(np.sum((t==(approx>.5)))/len(t))
        
    _, axs = plt.subplots(1,3)

    axs[0].set_title('weights')
    axs[0].plot([w[0] for w in w_values], label = "w0")
    axs[0].plot([w[1] for w in w_values], label = "w1")
    axs[0].plot([w[2] for w in w_values], label = "w2")
    axs[0].legend()

    axs[1].set_title('weights')
    # TODO plot most recent hessian as vairance
    axs[1].plot([w[1] for w in w_values], [w[2] for w in w_values])
    axs[1].scatter([w[1] for w in w_values], [w[2] for w in w_values])
    axs[1].set_xlabel('w1')
    axs[1].set_ylabel('w2')

    axs[2].set_title('accuracy')
    axs[2].plot(scores)
    axs[2].set_ylim([0, 1])

    # fig.tight_layout(pad = 1.5)
    plt.show()


if __name__ == '__main__':
    main()
