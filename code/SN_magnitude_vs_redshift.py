### matplotlib package -- https://matplotlib.org/stable/index.html ###
from matplotlib import pyplot as plt    #   for plotting 

### numpy package -- https://numpy.org/doc/stable/ ###
import numpy as np                      #   for general scientific computation

### scipy package -- https://docs.scipy.org/doc/scipy/index.html ###
# from scipy import constants as const    #   for physical constants -- https://docs.scipy.org/doc/scipy/reference/constants.html 
from scipy.integrate import quad          #   for integration -- https://docs.scipy.org/doc/scipy/tutorial/integrate.html
# from scipy import optimize as opt       #   for optimization and fit -- https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
# from scipy import special as sp         #   for special mathematical functions -- https://docs.scipy.org/doc/scipy/reference/tutorial/special.html

plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = r'''
# '''


def integrand(x, Omega_r0, Omega_m0, Omega_Lambda0):
    if x == 0.0:
        return 1.0
    Omega_K0 = 1.0 - Omega_r0 - Omega_m0 - Omega_Lambda0

    # define one_x := 1.0 + x and replace higher powers of it by recursive multiplication of one_x, since '**'-operator may be slow
    one_x = 1.0 + x
    one_x2 = one_x * one_x
    one_x3 = one_x2 * one_x
    one_x4 = one_x2 * one_x2
    return 1.0/np.sqrt(Omega_r0*one_x4 + Omega_m0*one_x3 + Omega_K0*one_x2 + Omega_Lambda0)


def integral(z, Omega_r0, Omega_m0, Omega_Lambda0):
    # d_C/d_H = Integrate[1/E(z'), {z',0,z}]
    return quad(integrand, 0.0, z, args=(Omega_r0, Omega_m0, Omega_Lambda0))[0]


def distances(z, Omega_r0, Omega_m0, Omega_Lambda0):
    # Cosmological Parameters
    # =======================
    c = 299792.458
    h = 0.6736
    H_0 = h*100.0
    d_H = c/H_0
    # =======================

    I = np.array([integral(zi, Omega_r0, Omega_m0, Omega_Lambda0) for zi in z])

    Omega_K0 = 1.0 - Omega_r0 - Omega_m0 - Omega_Lambda0
    if Omega_K0 > 0.0:
        transverse_comoving_distance = d_H*1.0/np.sqrt(Omega_K0)*np.sinh(np.sqrt(Omega_K0)*I)

    elif Omega_K0 == 0.0:
        transverse_comoving_distance = d_H*I

    elif Omega_K0 < 0.0:
        transverse_comoving_distance = d_H*1.0/np.sqrt(abs(Omega_K0))*np.sin(np.sqrt(abs(Omega_K0))*I)

    angular_diameter_distance = 1/(1.0 + z)*transverse_comoving_distance
    luminosity_distance = (1.0 + z)*transverse_comoving_distance

    # return transverse_comoving_distance, angular_diameter_distance, luminosity_distance
    return luminosity_distance

def relative_magnitude(absolute_magnitude, luminosity_distance):
    # luminosity_distance per Mpc, absolute_magnitude is at 10 pc (therefore + 25.0 since 10 pc = 10**(-5.0) Mpc)
    return absolute_magnitude + 5.0*np.log10(luminosity_distance) + 25.0


def main():    
    # Import data

    DATA_DIR = '../data/SN-data.txt'
    names = np.loadtxt(DATA_DIR, dtype = 'str', skiprows = 5, usecols = 0)
    redshifts = np.loadtxt(DATA_DIR, skiprows = 5, usecols = 1)
    magnitudes = np.loadtxt(DATA_DIR, skiprows = 5, usecols = 2)
    error_magnitudes = np.loadtxt(DATA_DIR, skiprows = 5, usecols = 3)
   
    z_min = min(redshifts)
    z_max = max(redshifts)
    z = np.linspace(z_min, z_max, 1000)

    fig, ax = plt.subplots()   
    
    plt.errorbar(redshifts, magnitudes, yerr = error_magnitudes, fmt = '.', elinewidth = 1, alpha = 0.4) 
   
   
    # magnitude vs. redshift for several cosmologies
    # ==============================================
   
    # Only matter (Einstein-de-Sitter universe)
    # -----------------------------------------
    Omega_r0 = 0.0
    Omega_m0 = 1.0
    Omega_Lambda0 = 0.0

    d_L = distances(z, Omega_r0, Omega_m0, Omega_Lambda0)
    m = relative_magnitude(0.0, d_L)
    
    plt.plot(z, m, color = 'blue', label = '$\\Omega_{m,0} = %.1f$, $\\Omega_{\\Lambda,0} = %.1f$ (Einstein-de-Sitter)'%(Omega_m0, Omega_Lambda0))
    # -----------------------------------------
   
    # Equilibrium between matter and Lambda
    # -------------------------------------
    Omega_r0 = 0.0
    Omega_m0 = 0.5
    Omega_Lambda0 = 0.5
    
    d_L = distances(z, Omega_r0, Omega_m0, Omega_Lambda0)
    m = relative_magnitude(0.0, d_L)
    
    plt.plot(z, m, color = 'purple', label = '$\\Omega_{m,0} = %.1f$, $\\Omega_{\\Lambda,0} = %.1f$ (matter-$\\Lambda$-equilibrium)'%(Omega_m0, Omega_Lambda0)) 
    # -------------------------------------

    # Todays values
    # -------------
    # Omega_m0 = 0.3153
    # Omega_Lambda0 = 0.6847
    # z_eq = 3402.0
    # Omega_r0 = 1.0/(1.0 + z_eq)*Omega_m0
    Omega_r0 = 0.0
    Omega_m0 = 0.3
    Omega_Lambda0 = 0.7
    
    d_L = distances(z, Omega_r0, Omega_m0, Omega_Lambda0)
    m = relative_magnitude(0.0, d_L)
    
    # plt.plot(z, m, color = 'red', label = '$\\Omega_r0 = %.4f$, $\\Omega_{m,0} = %.4f$, $\\Omega_{\\Lambda,0} = %.4f$ (todays values)'%(Omega_r0, Omega_m0, Omega_Lambda0))
    plt.plot(z, m, color = 'red', label = '$\\Omega_{m,0} = %.1f$, $\\Omega_{\\Lambda,0} = %.1f$ (todays values)'%(Omega_m0, Omega_Lambda0))
    # -------------    
    
    # Lambda dominant
    # ---------------
    Omega_r0 = 0.0
    Omega_m0 = 0.1
    Omega_Lambda0 = 0.9
    
    d_L = distances(z, Omega_r0, Omega_m0, Omega_Lambda0)
    m = relative_magnitude(0.0, d_L)
    
    plt.plot(z, m, color = 'orange', label = '$\\Omega_{m,0} = %.1f$, $\\Omega_{\\Lambda,0} = %.1f$ ($\\Lambda$-dominant)'%(Omega_m0, Omega_Lambda0))
    # ---------------

    
    plt.title('Apparent magnitude vs. redshift $m(z)$ for Type Ia supernovae')
    plt.xlabel('redshift $z$')
    plt.ylabel('magnitudes $m$')
    plt.legend(loc = 'upper left')
    plt.grid(True)
   
    plt.show()

    # fig.savefig('../thesis/figures/plots/EPS/SN_magnitude_vs_redshift.eps', format = 'eps', bbox_inches = 'tight')
    fig.savefig('../thesis/figures/plots/PNG/SN_magnitude_vs_redshift.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    # fig.savefig('../thesis/figures/plots/PDF/SN_magnitude_vs_redshift.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/SN_magnitude_vs_redshift.tex')

if __name__ == "__main__":
    main()
