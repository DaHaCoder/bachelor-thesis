### itertools package -- https://docs.python.org/3/library/itertools.html ###
from itertools import product                                               #   for creating a matrix with nrows*ncols -- https://docs.python.org/3/library/itertools.html#itertools.product

### matplotlib package -- https://matplotlib.org/stable/index.html ###
from matplotlib import pyplot as plt                                 #   for plotting
from mpl_toolkits.mplot3d import axes3d                              #   for plotting in 3d
from matplotlib.offsetbox import AnchoredText                        #   for text annotations in plots

### multiprocessing package -- https://docs.python.org/3/library/multiprocessing.html ###
from multiprocessing import Pool                                                        #   for faster computation

### numba package -- https://numba.pydata.org/numba-doc/latest/index.html ###
from numba import njit                                                      #   for faster code compilation ('jit' = just-in-time) -- https://numba.readthedocs.io/en/stable/user/5minguide.html

### numpy package -- https://numpy.org/doc/stable/ ###
import numpy as np                                   #   for general scientific computation

### scipy package -- https://docs.scipy.org/doc/scipy/index.html ###
from scipy.integrate import quad, dblquad                          #   for integration -- https://docs.scipy.org/doc/scipy/tutorial/integrate.html

### time package -- https://docs.python.org/3/library/time.html ###
import time                                                       #   for calculating computation time

### tikzplotlib package -- https://tikzplotlib.readthedocs.io/ ###
# import tikzplotlib

plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.size'] = 16
plt.rcParams['text.usetex'] = True                                                                          
plt.rcParams['text.latex.preamble'] = r'''
\usepackage{physics}                                                                    
'''


def timeit(message):
    def outer_wrapper(func):
        def inner_wrapper(*args, **kwargs):
            print(f"{message} ...", end="", flush=True)
            start = time.perf_counter()
            ret = func(*args, **kwargs)
            end = time.perf_counter()
            print(f"done! (took {end-start:.2f} seconds)")
            return ret
        return inner_wrapper
    return outer_wrapper


# Cosmological functions #
# ====================== #
@njit
def expansion_function(z, Omega_m0):
    return np.sqrt(Omega_m0 * np.power(1.0 + z, 3) + 1.0 - Omega_m0)


@njit
def integrand(z, Omega_m0):
    if z == 0.0:
        return 1.0
    E = expansion_function(z, Omega_m0)
    return 1.0/E


def integral(z, Omega_m0):
    # d_C/d_H = Integrate[1/E(z'), {z',0,z}]
    return quad(integrand, 0.0, z, args=(Omega_m0))[0]


def mod_luminosity_distance(z, Omega_m0):
    # Cosmological Parameters
    # =======================
    c = 299792.458           # speed of light in vacuum in km/s
    # h = 0.6766              # Planck Collaboration 2018, Table 7, Planck+BAO -- https://www.aanda.org/articles/aa/full_html/2020/09/aa33880-18/T7.html
    # H_0 = h*100.0           # hubble constant in km/s per Mpc
    H_0 = 1.0                # dependence on hubble constant is set into the mod_absolute_magnitude, see theoretical_magnitude
    d_H = c/H_0              # hubble distance
    # =======================
    
    I = np.array([integral(zi, Omega_m0)  for zi in z])
   
    return (1.0 + z) * d_H * I


@njit
def theoretical_magnitude(mod_absolute_magnitude, mod_luminosity_distance):
    # mod_luminosity_distance := H_0 * luminosity_distance
    # mod_absolute_magnitude := absolute_magnitude - 5.0 * np.log10(H_0) + 25.0
    return mod_absolute_magnitude + 5.0 * np.log10(mod_luminosity_distance) 
# ====================== #


@njit
def analytic_chi_square(magnitudes, error_magnitudes, theoretical_magnitudes):
    c1 = 0.0
    b0 = 0.0
    b1 = 0.0
    for m, sigma, m_th in zip(magnitudes, error_magnitudes, theoretical_magnitudes):
        c1 += 1.0/(sigma * sigma)
        b0 += (m_th - m)/(sigma * sigma)
        b1 += ((m_th - m)/sigma) * ((m_th - m)/sigma)
    return b1 - b0 * b0/c1


def analytic_likelihood(Omega_m0, redshifts, magnitudes, error_magnitudes, L0, zero_NaNs=False):
    mod_absolute_magnitude = 0.0
    D_L = mod_luminosity_distance(redshifts, Omega_m0)
    m_th = theoretical_magnitude(mod_absolute_magnitude, D_L)
    chi_2 = analytic_chi_square(magnitudes, error_magnitudes, m_th)
    L = L0 * np.exp(-0.5 * chi_2)
    if np.isnan(L) and zero_NaNs:
        L = 0.0
    return L


@timeit("Compute normalization factor")
def normalization_factor(redshifts, magnitudes, error_magnitudes, guess=1.0, bounds_Omega_m0=(0.0, 1.0)):     
    L = lambda Omega_m0: analytic_likelihood(Omega_m0, redshifts, magnitudes, error_magnitudes, L0=guess, zero_NaNs=True)
    L0 = guess / quad(L, *bounds_Omega_m0)[0]
    return L0


def find_best_fit_values(LIST_analytic_likelihood):
    return np.nanargmax(LIST_analytic_likelihood)


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



    # === Computation of analytic_likelihood, finding best values for Omega_m0 and Omega_Lambda ===
    # =============================================================================================

    # --- define variables ---
    LIST_Omega_m0 = np.linspace(0.0, 1.0, 1000)
    
    # --- compute normalization factor ---
    # L0 = normalization_factor(redshifts, magnitudes, error_magnitudes, guess=1e+124)
    L0 = 2.4960339657594705e+123   
    
    print()
    print("====================")
    print(f"L0 = {L0}")
    print("====================")
    print()

    # --- compute analytic_likelihood with L0 = 1.0 for every value in LIST_Omega_m0---
    LIST_analytic_likelihood_L0_1 = [analytic_likelihood(Omega_m0, redshifts, magnitudes, error_magnitudes, 1.0) for Omega_m0 in LIST_Omega_m0]
    
    # --- find index of the value for Omega_m0 at which analytic_likelihood_L0_1 has its maximum ---
    Omega_m0_L0_1_index = find_best_fit_values(LIST_analytic_likelihood_L0_1)
    Omega_m0_L0_1_best = LIST_Omega_m0[Omega_m0_L0_1_index]
    Omega_Lambda0_L0_1_best = 1.0 - Omega_m0_L0_1_best
    # analytic_likelihood_L0_1_best = LIST_analytic_likelihood_L0_1[Omega_m0_L0_1_index]

    # # --- print Omega_m0_best and Omega_Lambda0_best ---
    # print("==================================")
    # print("Cosmological Parameter Estimation: ")
    # print("----------------------------------")
    # print(f"Omega_m0      = {Omega_m0_best:.3f}")
    # print(f"Omega_Lambda0 = {Omega_Lambda0_best:.3f}")
    # print("==================================")
    
    # --- compute analytic_likelihood with for every value in LIST_Omega_m0---
    LIST_analytic_likelihood = [analytic_likelihood(Omega_m0, redshifts, magnitudes, error_magnitudes, L0) for Omega_m0 in LIST_Omega_m0]

    # --- find index of the value for Omega_m0 at which analytic_likelihood has its maximum ---
    Omega_m0_index = find_best_fit_values(LIST_analytic_likelihood)
    Omega_m0_best = LIST_Omega_m0[Omega_m0_index]
    Omega_Lambda0_best = 1.0 - Omega_m0_best
    # analytic_likelihood_best = LIST_analytic_likelihood[Omega_m0_index]

    # --- print Omega_m0_best and Omega_Lambda0_best ---
    print("==================================")
    print("Cosmological Parameter Estimation: ")
    print("----------------------------------")
    print(f"Omega_m0      = {Omega_m0_best:.3f}")
    print(f"Omega_Lambda0 = {Omega_Lambda0_best:.3f}")
    print("==================================")
    # ====================================================================================


    # === Plot Omega_m0 vs. analytic_likelihood_L0_1 ===
    # ==================================================

    # --- plot ---
    fig, ax = plt.subplots()

    plt.plot(LIST_Omega_m0, LIST_analytic_likelihood_L0_1, color='tab:blue')
    # plt.plot(Omega_m0_L0_1_best, analytic_likelihood_L0_1_best, 'o', color='tab:red')
    plt.xlabel(r'$\Omega_{\text{m},0}$', fontsize=16)
    plt.ylabel(r'$L_{\text{A}}(\Omega_{\text{m},0} \vert D)$ with $L_{0} = 1$', fontsize=16)
    # plt.suptitle(r'$\texttt{MWE-analytic-likelihood.py}$', fontsize=20)
    ax.tick_params(labelsize=14)
    plt.grid(True)

    at = AnchoredText(fr'$\vb*{{\theta_{{\text{{best}}}}}} = ({Omega_m0_L0_1_best:.2f}, {Omega_Lambda0_L0_1_best:.2f})$', loc='upper right', borderpad=0.5, prop=dict(fontsize=16))
    at.patch.set(boxstyle='round,pad=0.2', fc='w', ec='0.5', alpha=0.9) 
    ax.add_artist(at)

    # plt.show()

    # --- save fig ---
    fig.savefig('../thesis/figures/plots/EPS/MWE-analytic-likelihood_L0_1.eps', format='eps', bbox_inches='tight')
    fig.savefig('../thesis/figures/plots/PNG/MWE-analytic-likelihood_L0_1.png', format='png', bbox_inches='tight', dpi=250)
    fig.savefig('../thesis/figures/plots/PDF/MWE-analytic-likelihood_L0_1.pdf', format='pdf', bbox_inches='tight')
    # tikzplotlib.save('../thesis/figures/tikz/MWE-analytic-likelihood_L0_1.tex')
    # ==================================================


    # === Plot Omega_m0 vs. analytic_likelihood ===
    # =============================================

    # --- plot ---
    fig, ax = plt.subplots()

    plt.plot(LIST_Omega_m0, LIST_analytic_likelihood, color='tab:blue')
    # plt.plot(Omega_m0_best, analytic_likelihood_best, 'o', color='tab:red')
    plt.xlabel(r'$\Omega_{\text{m},0}$', fontsize=16)
    plt.ylabel(r'$L_{\text{A}}(\Omega_{\text{m},0} \vert D)$', fontsize=16)
    # plt.suptitle(r'$\texttt{MWE-analytic-likelihood.py}$', fontsize=20)
    ax.tick_params(labelsize=14)
    plt.grid(True)

    at = AnchoredText(fr'$\vb*{{\theta_{{\text{{best}}}}}} = ({Omega_m0_best:.2f}, {Omega_Lambda0_best:.2f})$', loc='upper right', borderpad=0.5, prop=dict(fontsize=16))
    at.patch.set(boxstyle='round,pad=0.2', fc='w', ec='0.5', alpha=0.9) 
    ax.add_artist(at)

    # plt.show()

    # --- save fig ---
    fig.savefig('../thesis/figures/plots/EPS/MWE-analytic-likelihood.eps', format='eps', bbox_inches='tight')
    fig.savefig('../thesis/figures/plots/PNG/MWE-analytic-likelihood.png', format='png', bbox_inches='tight', dpi=250)
    fig.savefig('../thesis/figures/plots/PDF/MWE-analytic-likelihood.pdf', format='pdf', bbox_inches='tight')
    # tikzplotlib.save('../thesis/figures/tikz/MWE-analytic-likelihood.tex')
    # ==========================================================


if __name__ == "__main__":
    main()
