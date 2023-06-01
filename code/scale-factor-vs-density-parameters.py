### matplotlib package -- https://matplotlib.org/stable/index.html ###
from matplotlib import pyplot as plt                                 #   for plotting 

### numpy package -- https://numpy.org/doc/stable/ ###
import numpy as np                                   #   for general scientific computation

# ### tizplotlib package -- https://github.com/nschloe/tikzplotlib ###
# import tikzplotlib                                                 #    to save plots as .tex-file with tikz

plt.rcParams['font.family']='serif'
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'''
\usepackage{physics}
'''

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

    a0 = 1.0
    a = np.linspace(0.0, a0, int(1e7))
    
    fig, ax = plt.subplots()

    # Omega_r0 = 8.5*10**(-5.0)
    Omega_m0 = 0.3111 
    # a_eq = Omega_r0/Omega_m0, a_eq = 1.0/(1.0 + z_eq) --> Omega_r0 = a_eq*Omega_m0
    z_eq = 3387.0
    Omega_r0 = 1.0/(1.0 + z_eq)*Omega_m0
    Omega_Lambda0 = 0.6889
    # Omega_K0 = 0.0    
    Omega_K0 = 1.0 - Omega_r0 - Omega_m0 - Omega_Lambda0

    E_0_square = expansion_rate_squared(a, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0)

    plt.plot(a, Omega_r(a, Omega_r0, E_0_square), color='orange', label='$\\Omega_{\\text{r}}(a)$')
    plt.plot(a, Omega_m(a, Omega_m0, E_0_square), color='red', label='$\\Omega_{\\text{m}}(a)$') 
    plt.plot(a, Omega_Lambda(a, Omega_Lambda0, E_0_square), color='purple', label='$\\Omega_{\\Lambda}(a)$')


    ax.set_xscale("log", base = 10)

    # ax.set_xlim(0.0, 1.0)
    # ax.set_ylim(0.0, 1.0)

    plt.xlabel('scale factor $a(t)$')
    plt.ylabel('density parameters $\\Omega_{i}(a)$')
    plt.legend(loc='center left')
    plt.grid(True)

    # plt.show()

    fig.savefig('../thesis/figures/plots/EPS/scale-factor-vs-density-parameters.eps', format = 'eps', bbox_inches = 'tight')
    fig.savefig('../thesis/figures/plots/PNG/scale-factor-vs-density-parameters.png', format = 'png', bbox_inches = 'tight', dpi = 400)
    fig.savefig('../thesis/figures/plots/PDF/scale-factor-vs-density-parameters.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/plots/scale-factor-vs-density-parameters.tex')


if __name__ == "__main__":
    main()
