### matplotlib package -- https://matplotlib.org/stable/index.html ###
from matplotlib import pyplot as plt                                 #   for plotting 
from mpl_toolkits.mplot3d import axes3d                              #   for plotting in 3d

### numpy package -- https://numpy.org/doc/stable/ ###
import numpy as np                                   #   for general scientific computation

### scipy package -- https://docs.scipy.org/doc/scipy/index.html ###
# from scipy import constants as const                             #   for physical constants -- https://docs.scipy.org/doc/scipy/reference/constants.html 
from scipy.integrate import quad, dblquad                          #   for integration -- https://docs.scipy.org/doc/scipy/tutorial/integrate.html
# from scipy import optimize as opt                                #   for optimization and fit -- https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
# from scipy import special as sp                                  #   for special mathematical functions -- https://docs.scipy.org/doc/scipy/reference/tutorial/special.html

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


def distances(z, Omega_r0, Omega_m0, Omega_Lambda0, c, H_0):
    I = np.array([integral(zi, Omega_r0, Omega_m0, Omega_Lambda0) for zi in z])
    Omega_K0 = 1.0 - Omega_r0 - Omega_m0 - Omega_Lambda0
    
    d_H = c/H_0
    if Omega_K0 > 0.0:
        proper_motion_distance = d_H*1.0/np.sqrt(Omega_K0)*np.sinh(np.sqrt(Omega_K0)*I)

    elif Omega_K0 == 0.0:
        proper_motion_distance = d_H*I

    elif Omega_K0 < 0.0:
        proper_motion_distance = d_H*1.0/np.sqrt(abs(Omega_K0))*np.sin(np.sqrt(abs(Omega_K0))*I)

    angular_diameter_distance = 1.0/(1.0 + z)*proper_motion_distance
    luminosity_distance = (1.0 + z)*proper_motion_distance

    return proper_motion_distance, angular_diameter_distance, luminosity_distance


# def relative_magnitude(luminosity_distance, absolute_magnitude):
#     # luminosity_distance per Mpc, absolute_magnitude is at 10 pc (therefor + 25.0 since 10 pc = 10**(-5.0) Mpc)
#     return absolute_magnitude + 5.0*np.log10(luminosity_distance) + 25.0


def relative_magnitude(luminosity_distance, H_0, new_absolute_magnitude):
    # new_absolute_magnitude := absolute_magnitude - 5*np.log10(hubble_constant) + 25.0 
    return new_absolute_magnitude + 5*np.log10(H_0*luminosity_distance)


def chi_square_analytic(redshifts, magnitudes, error_magnitudes, relative_magnitudes):
    c1 = 0.0
    f0 = 0.0
    f1 = 0.0
    for z, m, sigma, mag in zip(redshifts, magnitudes, error_magnitudes, relative_magnitudes):
        c1 += 1.0/sigma**2.0
        f0 += (mag - m)/sigma**2.0
        f1 += ((mag - m)/sigma)**2.0
    return f1 - f0**2.0/c1 


# def likelihood(Omega_r0, LIST_Omega_m0, LIST_Omega_Lambda0, c, H_0, redshifts, magnitudes, error_magnitudes):
#     MATRIX = []
#     j = 0
#     for j in range(len(LIST_Omega_Lambda0)):
#         Omega_Lambda0 = LIST_Omega_Lambda0[j]
#         ROW = []
#         i = 0
#         for i in range(len(LIST_Omega_m0)):
#             Omega_m0 = LIST_Omega_m0[i]
            
#             d_L = distances(redshifts, Omega_r0, Omega_m0, Omega_Lambda0, c, H_0)[2]
#             absolute_magnitude = 0.0
#             mag = relative_magnitude(d_L, H_0, absolute_magnitude)
#             # mag = relative_magnitude(d_L, absolute_magnitude)

#             chi_2 = chi_square_analytic(redshifts, magnitudes, error_magnitudes, mag) 
#             print("===========")
#             print("chi_2 = ", chi_2)
#             L = 10**(123.0)*np.exp(-0.5*chi_2)
#             print("============")
#             print("L = ", L)

#             ROW.append(L)
#         MATRIX.append(ROW)
#     return MATRIX


def likelihood(Omega_r0, Omega_m0, Omega_Lambda0, c, H_0, redshifts, magnitudes, error_magnitudes):
    d_L = distances(redshifts, Omega_r0, Omega_m0, Omega_Lambda0, c, H_0)[2]
    absolute_magnitude = 0.0
    mag = relative_magnitude(d_L, H_0, absolute_magnitude)
    chi_2 = chi_square_analytic(redshifts, magnitudes, error_magnitudes, mag) 
    # print("===========")
    # print("chi_2 = ", chi_2)
    L = np.exp(-0.5*chi_2)
    # print("============")
    # print("L = ", L)
    return L 
    

def normalization_factor(Omega_r0, LIST_Omega_m0, LIST_Omega_Lambda0, c, H_0, redshifts, magnitudes, error_magnitudes):
    L = lambda Omega_m0, Omega_Lambda0: likelihood(Omega_r0, Omega_m0, Omega_Lambda0, c, H_0, redshifts, magnitudes, error_magnitudes)
    x_1 = lambda Omega_Lambda0: min(LIST_Omega_m0)
    x_2 = lambda Omega_Lambda0: max(LIST_Omega_m0)
    y_1 = min(LIST_Omega_Lambda0)
    y_2 = max(LIST_Omega_Lambda0)
    L0 = dblquad(L, y_1, y_2, x_1, x_2)[0]
    # print("L0 = ", L0)
    return L0


def MATRIX_normalized_likelihood(Omega_r0, LIST_Omega_m0, LIST_Omega_Lambda0, c, H_0, redshifts, magnitudes, error_magnitudes):
    
    L0 = normalization_factor(Omega_r0, LIST_Omega_m0, LIST_Omega_Lambda0, c, H_0, redshifts, magnitudes, error_magnitudes)
    
    MATRIX = []
    j = 0
    for j in range(len(LIST_Omega_Lambda0)):
        Omega_Lambda0 = LIST_Omega_Lambda0[j]
        ROW = []
        i = 0
        for i in range(len(LIST_Omega_m0)):
            Omega_m0 = LIST_Omega_m0[i]
            L = 1.0/L0*likelihood(Omega_r0, Omega_m0, Omega_Lambda0, c, H_0, redshifts, magnitudes, error_magnitudes)
            ROW.append(L)
        MATRIX.append(ROW)
    return MATRIX


def find_best_fit_values(LIST_Omega_m0, LIST_Omega_Lambda0, MATRIX_likelihood):
    Omega_m0_index = [(index, row.index(np.max(MATRIX_likelihood))) for index, row in enumerate(MATRIX_likelihood) if np.max(MATRIX_likelihood) in row][0][1]
    Omega_Lambda0_index = [(index, row.index(np.max(MATRIX_likelihood))) for index, row in enumerate(MATRIX_likelihood) if np.max(MATRIX_likelihood) in row][0][0]

    Omega_m0_max_likelihood = LIST_Omega_m0[Omega_m0_index]
    Omega_Lambda0_max_likelihood = LIST_Omega_Lambda0[Omega_Lambda0_index]

    print("===============================")
    print("Values for maximum likelihood: ")
    print("Omega_m0 = ", Omega_m0_max_likelihood)
    print("Omega_Lambda0 = ", Omega_Lambda0_max_likelihood)
    print("===============================")

    return Omega_m0_max_likelihood, Omega_m0_max_likelihood


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
    Omega_r0 = 0.0
    # =======================

    LIST_Omega_m0 = np.arange(0.0, 1.0, 0.01)
    LIST_Omega_Lambda0 = np.arange(0.0, 1.0, 0.01)

    MATRIX_likelihood = MATRIX_normalized_likelihood(Omega_r0, LIST_Omega_m0, LIST_Omega_Lambda0, c, H_0, redshifts, magnitudes, error_magnitudes)

    find_best_fit_values(LIST_Omega_m0, LIST_Omega_Lambda0, MATRIX_likelihood)
    
    Z = np.array(MATRIX_likelihood)
    X, Y = np.meshgrid(LIST_Omega_m0, LIST_Omega_Lambda0)

    fig = plt.figure()
    ax = plt.axes(projection = '3d')

    ax.plot_wireframe(X, Y, Z, edgecolor = 'blue', alpha = 0.3)

    ax.contourf(X, Y, Z, zdir='z', offset=0, cmap='coolwarm')
    ax.contourf(X, Y, Z, zdir='x', offset=0, cmap='coolwarm')
    ax.contourf(X, Y, Z, zdir='y', offset=0, cmap='coolwarm')
   
    ax.set(xlabel = r'$\Omega_{m,0}$', ylabel = r'$\Omega_{\Lambda,0}$', zlabel = r'$L(\Omega_{m,0}, \Omega_{\Lambda,0})$')
    
    plt.show()
    
    # fig.savefig('../thesis/figures/plots/EPS/analytic_chi_square_likelihood.eps', format = 'eps', bbox_inches = 'tight')
    fig.savefig('../thesis/figures/plots/PNG/analytic_chi_square_likelihood.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    # fig.savefig('../thesis/figures/plots/PDF/analytic_chi_square_likelihood.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/analytic_chi_square_likelihood.tex')

if __name__ == "__main__":
    main()
