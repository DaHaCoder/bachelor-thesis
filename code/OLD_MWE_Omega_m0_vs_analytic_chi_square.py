### matplotlib package -- https://matplotlib.org/stable/index.html ###
from matplotlib import pyplot as plt        #   for plotting 
from mpl_toolkits.mplot3d import axes3d     #   for plotting in 3d

### numpy package -- https://numpy.org/doc/stable/ ###
import numpy as np                      #   for general scientific computation

### scipy package -- https://docs.scipy.org/doc/scipy/index.html ###
# from scipy import constants as const    #   for physical constants -- https://docs.scipy.org/doc/scipy/reference/constants.html 
from scipy.integrate import quad          #   for integration -- https://docs.scipy.org/doc/scipy/tutorial/integrate.html
# from scipy import optimize as opt       #   for optimization and fit -- https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
# from scipy import special as sp         #   for special mathematical functions -- https://docs.scipy.org/doc/scipy/reference/tutorial/special.html

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r''' 
\usepackage{amsmath, amsfonts, amssymb, amsthm}
'''


def integrand(x, Omega_m0):
    if x == 0.0:
        return 0.0
    return 1.0/np.sqrt(Omega_m0*(1.0+x)**(3.0) + (1.0 - Omega_m0))


def integral(z, Omega_m0):
    # D_C/D_H = Integrate[1/E(z'), {z',0,z}]
    return quad(integrand, 0.0, z, args=(Omega_m0))[0]


def luminosity_distance(z, Omega_m0, c, hubble_constant):
    I = np.array([integral(zi, Omega_m0) for zi in z])
    return (1.0 + z)*c/hubble_constant*I


def relative_magnitude(luminosity_distance, hubble_constant, new_absolute_magnitude):
    # new_absolute_magnitude := absolute_magnitude - 5*np.log10(hubble_constant) + 25.0 
    return new_absolute_magnitude + 5.0*np.log10(hubble_constant*luminosity_distance)


def chi_square_analytic(LIST_Omega_m0, c, hubble_constant, redshifts, magnitudes, error_magnitudes):
    LIST_chi_square_analytic = []
    z = redshifts       
    for Omega_m0 in zip(LIST_Omega_m0):
        d_L = luminosity_distance(z, Omega_m0, c, hubble_constant)
        absolute_magnitude = 0.0 
        rel_mag = relative_magnitude(d_L, hubble_constant, absolute_magnitude)

        c1 = 0.0
        f0 = 0.0
        f1 = 0.0

        for m, sigma, mag in zip(magnitudes, error_magnitudes, rel_mag):
            c1 += 1.0/sigma**2.0
            f0 += (mag - m)/sigma**2.0
            f1 += ((mag - m)/sigma)**2.0

        chi_2 = f1 - f0**2.0/c1
        LIST_chi_square_analytic.append(chi_2)
    
    return LIST_chi_square_analytic


def main():    
    # Import data

    DATA_DIR = '../data/SN-data.txt'
    names = np.loadtxt(DATA_DIR, dtype = 'str', skiprows = 5, usecols = 0)
    redshifts = np.loadtxt(DATA_DIR, skiprows = 5, usecols = 1)
    magnitudes = np.loadtxt(DATA_DIR, skiprows = 5, usecols = 2)
    error_magnitudes = np.loadtxt(DATA_DIR, skiprows = 5, usecols = 3)

    # Cosmological Parameters
    # =======================
    c = 299792.458
    h = 0.6736
    H_0 = h*100.0 
    # d_H = c/H_0
    # Omega_r0 = 0.0
    # =======================

    LIST_Omega_m0 = np.arange(0.0, 1.5, 0.01)

    LIST_chi_square_analytic = chi_square_analytic(LIST_Omega_m0, c, H_0, redshifts, magnitudes, error_magnitudes)
    
    index = LIST_chi_square_analytic.index(min(LIST_chi_square_analytic))
    Omega_m0_min = LIST_Omega_m0[index]
    
    print("=== === === ===")
    print("Omega_m0_min = ", Omega_m0_min)
    print("=== === === ===")
    
    fig, ax = plt.subplots()
    
    ax.plot(LIST_Omega_m0, LIST_chi_square_analytic, label = '$\Omega_{m,0,\\text{min}} = %.2f$'%(Omega_m0_min))
    ax.scatter(Omega_m0_min, min(LIST_chi_square_analytic), color = 'red')
  
    ax.set(xlabel = r'$\Omega_{m,0}$', ylabel = r'$\tilde{\chi}^2(\Omega_{m,0})$')
    plt.legend(loc = 'upper left')
    plt.grid(True)
    
    plt.show()

    # fig.savefig('../thesis/figures/plots/EPS/MWE_Omega_m0_vs_analytic_chi_square.eps', format = 'eps', bbox_inches = 'tight')
    fig.savefig('../thesis/figures/plots/PNG/MWE_Omega_m0_vs_analytic_chi_square.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    # fig.savefig('../thesis/figures/plots/PDF/MWE_Omega_m0_vs_analytic_chi_square.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/MWE_Omega_m0_vs_analytic_chi_square.tex')

if __name__ == "__main__":
    main()
