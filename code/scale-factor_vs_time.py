### matplotlib package -- https://matplotlib.org/stable/index.html ###
from matplotlib import pyplot as plt    #   for plotting 

### numpy package -- https://numpy.org/doc/stable/ ###
import numpy as np                      #   for general scientific computation

# ### scipy package -- https://docs.scipy.org/doc/scipy/index.html ###
# from scipy import constants as const    #   for physical constants -- https://docs.scipy.org/doc/scipy/reference/constants.html 
from scipy.integrate import quad          #   for integration -- https://docs.scipy.org/doc/scipy/tutorial/integrate.html
# from scipy import optimize as opt       #   for optimization and fit -- https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
# from scipy import special as sp         #   for special mathematical functions -- https://docs.scipy.org/doc/scipy/reference/tutorial/special.html

plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = r'''
# '''


def integrand(x, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0):
    if x == 0.0:
        return 0.0
    return 1.0/np.sqrt(Omega_r0*x**(-2.0) + Omega_m0*x**(-1.0) + Omega_K0 + Omega_Lambda0*x**(2.0))

def integral(a, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0):
    return quad(integrand, 0.0, a, args=(Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0))[0]

def main():

    a_max = 1.3
    a0 = 1.0
    # time_Hubble = 1.4*10**(10.0)

    a = np.linspace(0.0, a_max, 1000)
    
    fig, ax = plt.subplots()
    
    ##########
    
    # Flat universe (K = 0) without radiation (Omega_r = 0)
    # =====================================================

    Omega_r0 = 0.0
    Omega_K0 = 0.0

    # Only matter (Einstein-de-Sitter universe)
    Omega_m0 = 1.0
    Omega_Lambda0 = 0.0

    time_only_matter = np.array([integral(ai, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0) for ai in a])
    age_only_matter = integral(a0, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0)

    plt.plot(time_only_matter, a, color = 'blue', label = '$\\Omega_{m} = 1.0, \\Omega_{\\Lambda} = 0.0, t_{age} = %.2f$ (Einstein-de Sitter)'%age_only_matter)
    plt.vlines(age_only_matter, ymin = 0.0, ymax = a0, color = 'blue', alpha = 0.5, linestyle = '--')
    plt.hlines(a0, xmin = 0.0, xmax = age_only_matter, color = 'grey', linestyle = '--')

    # Equilibrium between matter and Lambda
    Omega_m0 = 0.5
    Omega_Lambda0 = 0.5 
   
    time_equilibrium = np.array([integral(ai, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0) for ai in a])
    age_equilibrium = integral(a0, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0)

    plt.plot(time_equilibrium, a, color = 'purple', label = '$\\Omega_{m} = 0.5, \\Omega_{\\Lambda} = 0.5, t_{age} = %.2f$ (matter-$\\Lambda$-equilibrium)'%age_equilibrium)
    plt.vlines(age_equilibrium, ymin = 0.0, ymax = a0, color = 'purple', alpha = 0.5, linestyle = '--')
    plt.hlines(a0, xmin = 0.0, xmax = age_equilibrium, color = 'grey', linestyle = '--')

    # todays values
    Omega_m0 = 0.3
    Omega_Lambda0 = 0.7 
   
    time_today = np.array([integral(ai, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0) for ai in a])
    age_today = integral(a0, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0)

    plt.plot(time_today, a, color = 'red', label = '$\\Omega_{m} = 0.3, \\Omega_{\\Lambda} = 0.7, t_{age} = %.2f$ (todays values)'%age_today)
    plt.vlines(age_today, ymin = 0.0, ymax = a0, color = 'red', alpha = 0.5, linestyle = '--')
    plt.hlines(a0, xmin = 0.0, xmax = age_today, color = 'grey', linestyle = '--')

    # Lambda dominant 
    Omega_m0 = 0.1
    Omega_Lambda0 = 0.9 
   
    time_Lambda_dominant = np.array([integral(ai, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0) for ai in a])
    age_Lambda_dominant = integral(a0, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0)

    plt.plot(time_Lambda_dominant, a, color = 'orange', label = '$\\Omega_{m} = 0.1, \\Omega_{\\Lambda} = 0.9, t_{age} = %.2f$ ($\\Lambda$-dominant)'%age_Lambda_dominant)
    plt.vlines(age_Lambda_dominant, ymin = 0.0, ymax = a0, color = 'orange', alpha = 0.5, linestyle = '--')
    plt.hlines(a0, xmin = 0.0, xmax = age_Lambda_dominant, color = 'grey', linestyle = '--')
   
    ##########

    ax.set_xlim(0.0, 1.5*age_today)
    ax.set_ylim(0.0, a_max)

    plt.title('Flat universe ($K = 0$) without radiation ($\\Omega_{r} = 0$)')
    plt.xlabel('$t/t_{H}$ with $t_{H} \\approx 14 \cdot 10^{9}$ years')
    plt.ylabel('scale factor $a$')
    plt.legend(loc = 'upper left')
    plt.grid(True)

    plt.show()

    # fig.savefig('../thesis/figures/plots/EPS/scale-factor_vs_time.eps', format = 'eps', bbox_inches = 'tight')
    fig.savefig('../thesis/figures/plots/PNG/scale-factor_vs_time.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    # fig.savefig('../thesis/figures/plots/PDF/scale-factor_vs_time.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/scale-factor_vs_time.tex')


if __name__ == "__main__":
    main()
