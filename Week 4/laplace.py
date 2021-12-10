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


def f(w):
    return np.dot(x, w)


def sigmoid(y):
    return 1/(1+np.exp(-y))


def G(w):
    y_list = sigmoid(f(w))

    result = 0
    for idx, y in enumerate(y_list):
        result -= t[idx]*np.log(y) + (1-t[idx])*np.log(1-y)
    return result


def M(w, alpha):
    return G(w) + .5*alpha*np.dot(w,w)


def hessian(w, alpha):
    _, d = x.shape
    h = f(w)

    hess = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            # print(np.dot(x, x.T).shape)
            # print(h.shape)
            hess[i,j] = np.sum( sigmoid(h) * sigmoid(-h) * np.dot(x, x.T) )

    return hess + alpha*np.identity(d)


def gradient_descent(w, alpha, learning_rate, iterations):  # TODO implement minimum
    e = 10**-10

    for _ in range(iterations):
        grad = (M(w+e, alpha)-M(w, alpha)) / e    # Simple
        w -= learning_rate * grad
    return w


def l_approx(w, alpha, epochs):
    w_values = [w]
    for _ in tqdm(range(epochs)):
        w_star = gradient_descent(w_values[-1], alpha, .005, 50)
        h = hessian(w_star, alpha)
        hinv = np.linalg.inv(h)
        w = np.random.multivariate_normal(w_star, hinv)
        w_values.append(w)

    return w_values


def main():
    np.random.seed(0)
    epochs = 35
    alpha = 0.5
    w = np.random.multivariate_normal([0,0,0], np.eye(3))

    w_values = l_approx(w, alpha, epochs)
    # print(x)
    # print(w_values[0])
    # print((w_values[0]*x))
    mse = [np.mean(np.square(np.sum((x*w), axis=1)-t)) for w in w_values]
    
    c = cm.rainbow(np.linspace(0, 1, len(w_values)))
    
    fig, axs = plt.subplots(1,3)

    axs[0].set_title('weights')
    axs[0].plot([w[0] for w in w_values], label = "w0")
    axs[0].plot([w[1] for w in w_values], label = "w1")
    axs[0].plot([w[2] for w in w_values], label = "w2")
    axs[0].legend()

    axs[1].set_title('weights')
    axs[1].plot([w[1] for w in w_values], [w[2] for w in w_values])
    axs[1].scatter([w[1] for w in w_values], [w[2] for w in w_values])
    axs[1].set_ylabel('w1')
    axs[1].set_ylabel('w2')
    # axs[1].set_ylim([-3,5])
    # axs[1].set_xlim([-1,7])

    axs[2].set_title('MSE')
    axs[2].plot(mse)

    # fig.tight_layout(pad = 1.5)
    plt.show()


if __name__ == '__main__':
    main()
