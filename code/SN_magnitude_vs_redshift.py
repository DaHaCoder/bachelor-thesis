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
    return 1.0/np.sqrt(Omega_r0*(1.0+x)**(4.0) + Omega_m0*(1.0+x)**(3.0) + Omega_K0*(1.0+x)**(2.0) + Omega_Lambda0)


def integral(z, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0):
    # D_C/D_H = Integrate[1/E(z'), {z',0,z}]
    return quad(integrand, 0.0, z, args=(Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0))[0]


def distances(z, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0, hubble_distance):
    I = np.array([integral(zi, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0) for zi in z])

    if Omega_K0 > 0.0:
        proper_motion_distance = hubble_distance*1.0/np.sqrt(Omega_K0)*np.sinh(np.sqrt(Omega_K0)*I)

    elif Omega_K0 == 0.0:
        proper_motion_distance = hubble_distance*I

    elif Omega_K0 < 0.0:
        proper_motion_distance = hubble_distance*1.0/np.sqrt(abs(Omega_K0))*np.sin(np.sqrt(abs(Omega_K0))*I)

    angular_diameter_distance = 1/(1.0 + z)*proper_motion_distance
    luminosity_distance = (1.0 + z)*proper_motion_distance

    return proper_motion_distance, angular_diameter_distance, luminosity_distance 

def magnitude(z, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0, hubble_distance, absolute_magnitude):
    # luminosity_distance per Mpc, absolute_Magnitude is at 10 pc (therefor + 25.0 since 10 pc = 10**(-5.0) Mpc)
    luminosity_distance = distances(z, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0, hubble_distance)[2] 
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
   
    # Cosmological Parameters
    # =======================
    c = 299792.458
    h = 0.6736
    H_0 = h*100.0 
    d_H = c/H_0
    # =======================
   

    # magnitude vs. redshift for several cosmologies
    # ==============================================
   
    # Only matter (Einstein-de-Sitter universe)
    # -----------------------------------------
    Omega_r0 = 0.0
    Omega_m0 = 1.0
    Omega_Lambda0 = 0.0
    # Omega_K0 = 0.0
    Omega_K0 = 1.0 - Omega_r0 - Omega_m0 - Omega_Lambda0

    m = magnitude(z, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0, d_H, 0.0)
    
    plt.plot(z, m, color = 'blue', label = '$\\Omega_{m,0} = %.1f$, $\\Omega_{\\Lambda,0} = %.1f$ (Einstein-de-Sitter)'%(Omega_m0, Omega_Lambda0))
    # -----------------------------------------
   
    # Equilibrium between matter and Lambda
    # -------------------------------------
    Omega_r0 = 0.0
    Omega_m0 = 0.5
    Omega_Lambda0 = 0.5
    # Omega_K0 = 0.0
    Omega_K0 = 1.0 - Omega_r0 - Omega_m0 - Omega_Lambda0
    
    m = magnitude(z, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0, d_H, 0.0)
    
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
    # Omega_K0 = 0.0
    Omega_K0 = 1.0 - Omega_r0 - Omega_m0 - Omega_Lambda0
    
    m = magnitude(z, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0, d_H, 0.0)
    
    # plt.plot(z, m, color = 'red', label = '$\\Omega_r0 = %.4f$, $\\Omega_{m,0} = %.4f$, $\\Omega_{\\Lambda,0} = %.4f$ (todays values)'%(Omega_r0, Omega_m0, Omega_Lambda0))
    plt.plot(z, m, color = 'red', label = '$\\Omega_{m,0} = %.1f$, $\\Omega_{\\Lambda,0} = %.1f$ (todays values)'%(Omega_m0, Omega_Lambda0))
    # -------------    
    
    # Lambda dominant
    # ---------------
    Omega_r0 = 0.0
    Omega_m0 = 0.1
    Omega_Lambda0 = 0.9
    # Omega_K0 = 0.0
    Omega_K0 = 1.0 - Omega_r0 - Omega_m0 - Omega_Lambda0
    
    m = magnitude(z, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0, d_H, 0.0)
    
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
