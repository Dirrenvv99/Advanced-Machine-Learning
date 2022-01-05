import numpy as np
from scipy import integrate
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
from sys import platform

#Loading of the data generated in previous steps
if platform == "linux" or platform == "linux2" or platform == "darwin":
    filename = "./q_and_m.json"
elif platform == "win32":
    filename = ".\q_and_m.json"


with open(filename) as f:
    data = json.load(f)
    qs = np.array(data["qs"])
    ms = np.array(data["ms"])

def filling():
    Hs = np.empty_like(qs)
    E_mean = np.empty_like(qs)
    xs = np.linspace(0.05,2,40)
    ys = np.linspace(0.05,2,40)

    def f(z,J_0,m,J,q):
        return J_0*m + J*np.sqrt(q)*z

    def h_func(z,J_0,m,J,q):
        # As within the assignment is mentioned. The original integral gives some problems
        # return np.exp(-z**(2)/2) * np.log(2 * np.cosh(J_0 * m + J * np.sqrt(q) * z))
        abs_value = np.abs(f(z,J_0,m,J,q))
        return np.exp(-z**(2)/2)*(abs_value + np.log(1 + np.exp(-2*abs_value)))


    for index_x,  x in enumerate(xs):
        for index_y, y in enumerate(ys):
            q = qs[len(ys) - 1 - index_y][index_x]
            m = ms[len(ys) - 1 - index_y][index_x]

            J = 1/y
            J_0 = x/y
            
            H_site = -J**(2)/4 * (q-1)**(2) - J_0*m**(2) - J**(2)*q*(1-q) + (1/np.sqrt(np.pi*2)) * integrate.quad(h_func, -np.inf, np.inf, args = (J_0,m,J,q))[0]
            E_mean_site = -0.5*J_0*m**(2) - 0.5*J**(2)*(1-q**(2))


            Hs[len(ys) - 1 - index_y][index_x] = H_site
            E_mean[len(ys) - 1 - index_y][index_x] = E_mean_site
    return Hs, E_mean

def main():
    Hs, E_mean = filling()

    fig, ax = plt.subplots(1,2)
    divider_0 = make_axes_locatable(ax[0])
    cax_0 = divider_0.append_axes('right', size='5%', pad=0.05)

    im_0 = ax[0].imshow(Hs, extent = [0,2,0,2])
    ax[0].set_title("H")
    ax[0].set_xlabel("J\N{SUBSCRIPT ZERO}/J")
    ax[0].set_ylabel(r"1/J")

    fig.colorbar(im_0, cax=cax_0, orientation='vertical')

    H_threshold = (Hs > 0).astype(int)

    divider_1 = make_axes_locatable(ax[1])
    cax_1 = divider_1.append_axes('right', size='5%', pad=0.05)

    im_1 = ax[1].imshow(H_threshold, extent = [0,2,0,2])   
    ax[1].set_title("H")
    ax[1].set_xlabel("J\N{SUBSCRIPT ZERO}/J")
    ax[1].set_ylabel(r"1/J")

    fig.colorbar(im_1, cax=cax_1, orientation='vertical')
    

    plt.show()

if __name__ == '__main__':
    main()


