import numpy as np
from scipy import integrate
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def q_func(z,x, y, m_0, q_0):
    J = 1/y
    J_0 = x/y
    return np.exp(-z**(2)/2)*(1/np.cosh(J*np.sqrt(q_0)*z + J_0 * m_0))**(2)

def m_func(z,x,y,m_0,q_0):
    J = 1/y
    J_0 = x/y
    return np.exp(-z**(2)/2)*(np.tanh(J*np.sqrt(q_0)*z + J_0*m_0))

def fixed_point_iteration():
    xs = np.linspace(0.05,2,40)
    ys = np.linspace(0.05,2,40)
    qs = np.empty((len(xs), len(ys)))
    ms = np.empty((len(xs), len(ys)))

    for index_x, x in enumerate(tqdm(xs)):
        for index_y, y in enumerate(ys):
            #Deze beginwaarden zijn belangrijk voor het resultaat (geen idee waarom). Deze waarde geven de goede plotjes. Verslag is toch niet
            #benodigd. Waarschijnlijk omdat deze waarde in het "midden" beginnen.
            q_old = 0.5
            m_old = 0.5
            q_new = np.inf
            m_new = np.inf
            epsilon = 0.0000005

            diff_q = np.inf
            diff_m = np.inf        

            while diff_q > epsilon and diff_m > epsilon:
                #Gekozen voor het gebruik van quad en niet quadrature omdat quad de mogelijkheid geeft tot oneindige integratiegrenzen
                q_new = 1 - 1/np.sqrt(2*np.pi) * integrate.quad(q_func, -np.inf, np.inf, args = (x,y,m_old,q_old))[0]
                m_new = 1/np.sqrt(2*np.pi) * integrate.quad(m_func, -np.inf, np.inf, args = (x,y,m_old,q_old))[0]

                diff_q = np.abs(q_new - q_old)
                diff_m = np.abs(m_new - m_old)

                q_old = q_new
                m_old = m_new
            #Deze rare "invul manier" is zodat de assen hetzelfde lopen als in de opgave wanneer je imshow gebruikt
            qs[len(ys) - 1 - index_y][index_x] = q_new
            ms[len(ys) - 1 - index_y][index_x]= m_new

    return qs, ms

def main(): 
    qs, ms = fixed_point_iteration()

 
    fig, ax = plt.subplots(1,2)
    divider_0 = make_axes_locatable(ax[0])
    cax_0 = divider_0.append_axes('right', size='5%', pad=0.05)

    im_0 = ax[0].imshow(qs, extent = [0,2,0,2])
    ax[0].set_title("q")
    ax[0].set_ylabel(r"1/J")
    ax[0].set_xlabel(r"J/J_0")

    fig.colorbar(im_0, cax=cax_0, orientation='vertical')

    divider_1 = make_axes_locatable(ax[1])
    cax_1 = divider_1.append_axes('right', size='5%', pad=0.05)

    im_1 = ax[1].imshow(ms, extent = [0,2,0,2])   
    ax[1].set_title("m")
    ax[1].set_ylabel(r"1/J")
    ax[1].set_xlabel(r"J/J_0")

    fig.colorbar(im_1, cax=cax_1, orientation='vertical')
    

    plt.show()
#Comment over de plotjes: De rare strepen in de plotjes zijn een gevolg van quad, wat subclusters gebruikt die adaptief zijn.
#Waarschijnlijk worden de clusters naar waardeverschil slim gekozen. Daarom lopen deze "lijnen" precies over de overgang waardes
#En gedragen zich daar dus net wat anders dan de rest wat die lijnen oplevert!
#Dit is opgelost door de epsilon wat strenger te maken. Ze zijn nu weg met een runtime van 3 minuten, dus helemaal top!
if __name__ == '__main__':
    main()
    

                
            
            




