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
# plt.rcParams['text.latex.preamble'] = r'''
# '''


def integrand(x, Omega_r0, Omega_m0, Omega_Lambda0):
    if x == 0.0:
        return 0.0
    Omega_K0 = 1.0 - Omega_r0 - Omega_m0 - Omega_Lambda0
    return 1.0/np.sqrt(Omega_r0*(1.0+x)**(4.0) + Omega_m0*(1.0+x)**(3.0) + Omega_K0*(1.0+x)**(2.0) + Omega_Lambda0)


def integral(z, Omega_r0, Omega_m0, Omega_Lambda0):
    # D_C/D_H = Integrate[1/E(z'), {z',0,z}]
    return quad(integrand, 0.0, z, args=(Omega_r0, Omega_m0, Omega_Lambda0))[0]


def distances(z, Omega_r0, Omega_m0, Omega_Lambda0, hubble_distance):
    I = np.array([integral(zi, Omega_r0, Omega_m0, Omega_Lambda0) for zi in z])
    Omega_K0 = 1.0 - Omega_r0 - Omega_m0 - Omega_Lambda0

    if Omega_K0 > 0.0:
        proper_motion_distance = hubble_distance*1.0/np.sqrt(Omega_K0)*np.sinh(np.sqrt(Omega_K0)*I)

    elif Omega_K0 == 0.0:
        proper_motion_distance = hubble_distance*I

    elif Omega_K0 < 0.0:
        proper_motion_distance = hubble_distance*1.0/np.sqrt(abs(Omega_K0))*np.sin(np.sqrt(abs(Omega_K0))*I)

    angular_diameter_distance = 1.0/(1.0 + z)*proper_motion_distance
    luminosity_distance = (1.0 + z)*proper_motion_distance

    return proper_motion_distance, angular_diameter_distance, luminosity_distance


def relative_magnitude(luminosity_distance, absolute_magnitude):
    # luminosity_distance per Mpc, absolute_magnitude is at 10 pc (therefor + 25.0 since 10 pc = 10**(-5.0) Mpc)
    return absolute_magnitude + 5.0*np.log10(luminosity_distance) + 25.0


def chi_square(redshifts, magnitudes, error_magnitudes, relative_magnitudes):
    SUM = 0.0
    for z, m, sigma, mag in zip(redshifts, magnitudes, error_magnitudes, relative_magnitudes):
        SUM += ((mag - m)/np.sqrt(2.0)*sigma)**2.0
    return SUM 


def likelihood(Omega_r0, LIST_Omega_m0, LIST_Omega_Lambda0, hubble_distance, redshifts, magnitudes, error_magnitudes):
    MATRIX = []
    j = 0
    for j in range(len(LIST_Omega_Lambda0)):
        Omega_Lambda0 = LIST_Omega_Lambda0[j]
        ROW = []
        i = 0
        for i in range(len(LIST_Omega_m0)):
            Omega_m0 = LIST_Omega_m0[i]
            
            z_min = min(redshifts)
            z_max = max(redshifts)
            z = np.linspace(z_min, z_max, 1000)
            
            d_L = distances(z, Omega_r0, Omega_m0, Omega_Lambda0, hubble_distance)[2]
            absolute_magnitude = 0.0
            mag = relative_magnitude(d_L, absolute_magnitude)

            chi_2 = chi_square(redshifts, magnitudes, error_magnitudes, mag) 
            L = np.exp(-chi_2)
            
            ROW.append(L)
        MATRIX.append(ROW)
    return MATRIX


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
    d_H = c/H_0
    # Omega_r0 = 0.0
    # =======================

    hubble_distance = d_H
    Omega_r0 = 0.0
    LIST_Omega_m0 = np.arange(0.0, 1.5, 0.1)
    LIST_Omega_Lambda0 = np.arange(-1.0, 2.0, 0.1)

    Z = np.array(likelihood(Omega_r0, LIST_Omega_m0, LIST_Omega_Lambda0, hubble_distance, redshifts, magnitudes, error_magnitudes))
    X, Y = np.meshgrid(LIST_Omega_m0, LIST_Omega_Lambda0)

    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    
    ax.plot_wireframe(X, Y, Z, edgecolor = 'blue', alpha = 0.3)
    
    ax.contourf(X, Y, Z, zdir='z', offset=0, cmap='coolwarm')
    ax.contourf(X, Y, Z, zdir='x', offset=0, cmap='coolwarm')
    ax.contourf(X, Y, Z, zdir='y', offset=0, cmap='coolwarm')
    
    ax.set(xlabel = r'$\Omega_{m,0}$', ylabel = r'$\Omega_{\Lambda,0}$', zlabel = r'$L(\Omega_{m,0}, \Omega_{\Lambda,0})$')
    
    plt.show()
    
    # fig.savefig('../thesis/figures/plots/EPS/likelihood_Omega_m0_vs_Omega_Lambda0.eps', format = 'eps', bbox_inches = 'tight')
    # fig.savefig('../thesis/figures/plots/PNG/likelihood_Omega_m0_vs_Omega_Lambda0.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    # fig.savefig('../thesis/figures/plots/PDF/likelihood_Omega_m0_vs_Omega_Lambda0.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/likelihood_Omega_m0_vs_Omega_Lambda0.tex')

if __name__ == "__main__":
    main()
