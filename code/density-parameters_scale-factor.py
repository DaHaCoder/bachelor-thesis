### matplotlib package -- https://matplotlib.org/stable/index.html ###
from matplotlib import pyplot as plt    #   for plotting 

### numpy package -- https://numpy.org/doc/stable/ ###
import numpy as np                      #   for general scientific computation

# ### scipy package -- https://docs.scipy.org/doc/scipy/index.html ###
# from scipy import constants as const    #   for physical constants -- https://docs.scipy.org/doc/scipy/reference/constants.html 
# from scipy import optimize as opt       #   for optimization and fit -- https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
# from scipy import special as sp         #   for special mathematical functions -- https://docs.scipy.org/doc/scipy/reference/tutorial/special.html

plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = r'''
# '''

def expansion_rate_squared(a, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0):
    return Omega_r0*a**(-4.0) + Omega_m0*a**(-3.0) + Omega_K0*a**(-2.0) + Omega_Lambda0

def Omega_r(a, Omega_r0, E_squared):
    return Omega_r0*a**(-4.0)/E_squared

def Omega_K(a, Omega_K0, E_squared):
    return Omega_K0*a**(-2.0)/E_squared
 
def Omega_m(a, Omega_m0, E_squared):
    return Omega_m0*a**(-3.0)/E_squared

def Omega_Lambda(a, Omega_Lambda0, E_squared):
    return Omega_Lambda0/E_squared

def main():

    a = np.linspace(0,1,int(1e7))
    
    fig, ax = plt.subplots()

    Omega_r0 = 8.5*10**(-5.0)
    Omega_m0 = 0.3
    Omega_K0 = 0.0
    Omega_Lambda0 = 0.7

    E_0_square = expansion_rate_squared(a, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0)

    # plt.plot(a, expansion_rate_squared(a, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0), color = 'blue', label = '$E^2(a)$')
    plt.plot(a, Omega_r(a, Omega_r0, E_0_square), color = 'orange', label = '$\\Omega_{r}(a)$')
    plt.plot(a, Omega_m(a, Omega_m0, E_0_square), color = 'red', label = '$\\Omega_{m}(a)$') 
    plt.plot(a, Omega_Lambda(a, Omega_Lambda0, E_0_square), color = 'purple', label = '$\\Omega_{\\Lambda}(a)$')


    ax.set_xscale("log", base = 10)

    # ax.set_xlim(0.0, 1.0)
    # ax.set_ylim(0.0, 1.0)

    plt.xlabel('$a$')
    plt.ylabel('$\\Omega_{i}$')
    plt.legend(loc = 'center left')
    plt.grid(True)

    # plt.show()

    # fig.savefig('../thesis/figures/plots/EPS/density-parameters_scale-factor.eps', format = 'eps', bbox_inches = 'tight')
    fig.savefig('../thesis/figures/plots/PNG/density-parameters_scale-factor.png', format = 'png', bbox_inches = 'tight', dpi = 400)
    # fig.savefig('../thesis/figures/plots/PDF/density-parameters_scale-factor.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/density-parameters_scale-factor.tex')


if __name__ == "__main__":
    main()
