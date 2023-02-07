### itertools package -- https://docs.python.org/3/library/itertools.html ###
from itertools import product                                               # for creating a matrix with nrows*ncols -- https://docs.python.org/3/library/itertools.html#itertools.product

### matplotlib package -- https://matplotlib.org/stable/index.html ###
from matplotlib import pyplot as plt                                 #   for plotting
from mpl_toolkits.mplot3d import axes3d                              #   for plotting in 3d

### multiprocessing package -- https://docs.python.org/3/library/multiprocessing.html ###
from multiprocessing import Pool                                                        #   for faster computation

### numba package -- https://numba.pydata.org/numba-doc/latest/index.html ###
from numba import njit                                                      #   for faster code compilation ('jit' = just-in-time) -- https://numba.pydata.org/numba-doc/latest/user/5minguide.html?highlight=njit

### numpy package -- https://numpy.org/doc/stable/ ###
import numpy as np                                   #   for general scientific computation

### scipy package -- https://docs.scipy.org/doc/scipy/index.html ###
# from scipy import constants as const                             #   for physical constants -- https://docs.scipy.org/doc/scipy/reference/constants.html
from scipy.integrate import quad, dblquad, tplquad                 #   for integration -- https://docs.scipy.org/doc/scipy/tutorial/integrate.html
from scipy import optimize as opt                                  #   for optimization and fit -- https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
# from scipy import special as sp                                  #   for special mathematical functions -- https://docs.scipy.org/doc/scipy/reference/tutorial/special.html

### time package -- https://docs.python.org/3/library/time.html ###
import time                                                       #     for calculating computation time

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size':16})
# plt.rcParams['text.latex.preamble'] = r'''
# '''

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        ret = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Time for {func.__name__}: {end - start} seconds")
        return ret
    return wrapper


@njit
def integrand(x, Omega_m0):
    if x == 0.0:
        return 1.0
    
    # define one_x := 1.0 + x and replace higher powers of it by recursive mulitiplications of one_x, since '**'-operator may be slow
    one_x = 1.0 + x
    one_x3 = one_x * one_x * one_x
    return 1.0/np.sqrt(Omega_m0*one_x3 + 1.0 - Omega_m0)


def integral(z, Omega_m0):
    # d_C/d_H = Integrate[1/E(z'), {z',0,z}]
    return quad(integrand, 0.0, z, args=(Omega_m0))[0]


def luminosity_distance(z, Omega_m0):
    # Cosmological Parameters
    # =======================
    c = 299792.458
    # h = 0.6736
    # H_0 = h*100.0
    H_0 = 1.0
    d_H = c/H_0
    # =======================
    
    I = np.array([integral(zi, Omega_m0)  for zi in z])
   
    return (1.0 + z) * d_H * I


@njit
def relative_magnitude(new_absolute_magnitude, new_luminosity_distance):
    # # luminosity_distance per Mpc, absolute_magnitude is at 10 pc (therefore + 25.0 since 10 pc = 10**(-5.0) Mpc)
    # return absolute_magnitude + 5.0*np.log10(luminosity_distance) + 25.0

    # h = 0.6736
    # H_0 = h*100.0
    # new_luminosity_distance := H_0 * luminosity_distance
    # new_absolute_magnitude := absolute_magnitude - 5.0 * np.log10(H_0) + 25.0
    return new_absolute_magnitude + 5.0 * np.log10(new_luminosity_distance) 


@njit
def chi_square_analytic(magnitudes, error_magnitudes, relative_magnitudes):
    c1 = 0.0
    f0 = 0.0
    f1 = 0.0
    for m, sigma, mag in zip(magnitudes, error_magnitudes, relative_magnitudes):
        c1 += 1.0/(sigma * sigma)
        f0 += (mag - m)/(sigma * sigma)
        f1 += ((mag - m)/sigma) * ((mag - m)/sigma)
    return f1 - f0 * f0/c1


# @njit
def likelihood(LIST_Omega_m0, redshifts, magnitudes, error_magnitudes):
    LIST_likelihood = []
    for i in range(len(LIST_Omega_m0)):
        Omega_m0 = LIST_Omega_m0[i]
        new_absolute_magnitude = 0.0 
        d_L = luminosity_distance(redshifts, Omega_m0)
        rel_mag = relative_magnitude(new_absolute_magnitude, d_L)
        chi_2 = chi_square_analytic(magnitudes, error_magnitudes, rel_mag)
        L = np.exp(-0.5 * chi_2)
        LIST_likelihood.append(L)
    return LIST_likelihood


def find_best_fit_values(LIST_likelihood):
    # FIXME: There are some NaN's in the matrix...
    return np.nanargmax(LIST_likelihood)


def gauss_curve(x, mu, sigma, y0):
    return y0*np.exp(-(x - mu) * (x - mu)/(2.0 * sigma * sigma))


def main():
    START_TOTAL_TIME = time.perf_counter()

    # Import data
    DATA_DIR = '../data/SN-data.txt'
    names, redshifts, magnitudes, error_magnitudes = np.loadtxt(DATA_DIR,
                                                                comments="#",
                                                                usecols=(0, 1, 2, 3),
                                                                dtype=np.dtype([("name", str),
                                                                                ("redshift", float),
                                                                                ("magnitude", float),
                                                                                ("sigma", float)]),
                                                                unpack=True)



    # === Computation of likelihood, finding best values for Omega_m0 and Omega_Lambda ===
    # ====================================================================================

    # --- define variables ---
    LIST_Omega_m0 = np.arange(0.0, 1.2, 0.001)

    # --- compute likelihood for every value in LIST_Omega_m0 and LIST_Omega_Lambda0 ---
    LIST_likelihood = likelihood(LIST_Omega_m0, redshifts, magnitudes, error_magnitudes)

    # --- find index of the values for Omega_m0 and Omega_Lambda0 at which the likelihood has its maximum ---
    Omega_m0_index = find_best_fit_values(LIST_likelihood)
    Omega_m0_best = LIST_Omega_m0[Omega_m0_index]

    Omega_Lambda0_best = 1.0 - Omega_m0_best

    # --- print best Omega_m0_best and Omega_Lambda0_best ---
    print("================================")
    print("Values for maximum likelihood: ")
    print("Omega_m0 = ", Omega_m0_best)
    print("Omega_Lambda0 = ", Omega_Lambda0_best)
    print("================================")
    # ====================================================================================



    # === Compute Omega_m0 vs. Omega_Lambda0 vs. likelihood ===
    # =========================================================

    # --- compute likelihood ---
    # Z = np.array(LIST_likelihood)
    # LIST_Omega_Lambda0 = 1.0 - np.array(LIST_Omega_m0)
    # X, Y = np.meshgrid(LIST_Omega_m0, LIST_Omega_Lambda0)

    # --- plot ---
    fig, ax = plt.subplots()

    plt.plot(LIST_Omega_m0, LIST_likelihood, color='tab:blue')
    plt.xlabel('$\Omega_{m,0}$')
    plt.ylabel('$L(\Omega_{m,0})$')
    plt.suptitle('$\\texttt{K0_analytic_likelihood.py}$', fontsize=20)
    plt.title('best fit values: $(\Omega_{m,0}, \Omega_{\Lambda,0}) = (%.2f, %.2f)$'%(Omega_m0_best, Omega_Lambda0_best))
    plt.grid(True)
    plt.show()

    # --- save fig ---
    # fig.savefig('../thesis/figures/plots/EPS/[K0_analytic_likelihood]_Omega_m0_vs_likelihood.eps', format = 'eps', bbox_inches = 'tight')
    fig.savefig('../thesis/figures/plots/PNG/[K0_analytic_likelihood]_Omega_m0_vs_likelihood.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    # fig.savefig('../thesis/figures/plots/PDF/[K0_analytic_likelihood]_Omega_m0_vs_likelihood.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/[K0_analytic_likelihood]_Omega_m0_vs_likelihood.tex')

    # # --- plot contour ---
    # fig, ax = plt.subplots()

    # ax.contourf(X, Y, Z, cmap='coolwarm')
    # plt.xlabel('$\Omega_{m,0}$')
    # plt.ylabel('$\Omega_{\Lambda,0}$')
    # plt.suptitle('$\\texttt{K0_analytic_likelihood.py}$', fontsize=20)
    # plt.title('best fit values: $(\Omega_{m,0}, \Omega_{\Lambda,0}) = (%.2f, %.2f)$'%(Omega_m0_best, Omega_Lambda0_best))
    # plt.grid(True)
    # plt.show()

    # # --- save fig ---
    # # fig.savefig('../thesis/figures/plots/EPS/[K0_analytic_likelihood]_Omega_m0_vs_Omega_Lambda0.eps', format = 'eps', bbox_inches = 'tight')
    # fig.savefig('../thesis/figures/plots/PNG/[K0_analytic_likelihood]_Omega_m0_vs_Omega_Lambda0.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    # # fig.savefig('../thesis/figures/plots/PDF/[K0_analytic_likelihood]_Omega_m0_vs_Omega_Lambda0.pdf', format = 'pdf', bbox_inches = 'tight')
    # # tikzplotlib.save('../thesis/figures/tikz/[K0_analytic_likelihood]_Omega_m0_vs_Omega_Lambda0.tex')
    # # =========================================================

if __name__ == "__main__":
    main()
