import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def p_star(x):
    return multivariate_normal.pdf(x, mean = [0,0], cov = [[250.25,-249.75],[-249.75,250.25]])

def sample_q(x_r, sigma):
    return np.random.multivariate_normal(x_r, sigma * np.eye(2), 1)[0]

def q(x,x_r,sigma):
    return multivariate_normal.pdf(x,mean = x_r, cov = sigma * np.eye(2))

def a_value(x_r, x_sample, sigma):

    nom = p_star(x_sample) * q(x_r,x_sample,sigma)
    denom = p_star(x_r) * q(x_sample, x_r, sigma)

    return nom/denom

def MH(sigma, iter):
    x = np.random.multivariate_normal(np.array([0,0]), np.eye(2))
    rejections = 0
    for _ in range(iter):
        sample = sample_q(x, sigma)
        a = a_value(x, sample, sigma)
        if a >= 1:
            x = sample
        else:
            if np.random.random() < a:
                x = sample
            else: 
                rejections +=1
    return x, rejections

def main():
    sigmas = np.linspace(0.2,600,10)
    mean_values = []
    rejection_rates = []
    fig, axs = plt.subplots(5,2, figsize = (18,15))
    real_samples = np.random.multivariate_normal([0,0], [[250.25,-249.75],[-249.75,250.25]], 100)
    for index, sigma in tqdm(enumerate(sigmas)):
        samples = []
        recs = []
        for _ in tqdm(range(100)):
            x_final, rejections = MH(sigma, 25)
            samples.append(x_final)
            recs.append(rejections)
        # x_plot = [x for x in np.linspace(-40,40,500)]
        # [X,Y] = np.meshgrid(x_plot,x_plot)
        # pos = np.empty(X.shape + (2,))
        # pos[:, :, 0] = X; pos[:, :, 1] = Y
        # p = multivariate_normal([0,0], [[250.25,-249.75],[-249.75,250.25]])
        # axs[index%5, int(index > 5)].contour(X,Y, p.pdf(pos))
        
        axs[index%5, int(index > 4)].scatter([x[0] for x in real_samples], [x[1] for x in real_samples], color = "blue", label = "normally sampled", marker = ".")
        axs[index%5, int(index > 4)].scatter([x[0] for x in samples], [x[1] for x in samples], color = "red", label = "samples")
        #confidence_ellipse(np.array([x[0] for x in samples]), np.array([x[1] for x in samples]), axs[index%5, int(index < 5)], edgecolor = "red")
        axs[index%5, int(index > 4)].set_title("sample plot with sigma: {:.2f}; rejection rate: {:.2f}".format(sigma, np.mean(rejections)/100))
        axs[index%5, int(index > 4)].legend()
    fig.tight_layout(pad = 3.0)
    plt.show()

if __name__ == '__main__':
    main()
