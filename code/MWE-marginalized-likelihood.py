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
# import tikzplotlib                                               #   for converting plot to tikz

plt.rcParams['font.family'] = 'serif' 
plt.rcParams['font.size'] = 16 
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

    I = np.array([integral(zi, Omega_m0) for zi in z])
   
    return (1.0 + z) * d_H * I 


@njit
def theoretical_magnitude(mod_absolute_magnitude, mod_luminosity_distance):
    # mod_luminosity_distance := H_0 * luminosity_distance
    # mod_absolute_magnitude := absolute_magnitude - 5.0 * np.log10(H_0) + 25.0
    return mod_absolute_magnitude + 5.0 * np.log10(mod_luminosity_distance) 
# ====================== #


@njit
def chi_square(magnitudes, error_magnitudes, theoretical_magnitudes):
    return np.sum(np.square((theoretical_magnitudes - magnitudes) / error_magnitudes))


@njit
def l(mod_absolute_magnitude, mod_luminosity_distance, magnitudes, error_magnitudes, L0):
    m_th = theoretical_magnitude(mod_absolute_magnitude, mod_luminosity_distance)
    chi_2 = chi_square(magnitudes, error_magnitudes, m_th)
    return L0 * np.exp(-0.5 * chi_2)


def marginalized_likelihood(Omega_m0, redshifts, magnitudes, error_magnitudes, L0, zero_NaNs=False):
    D_L = mod_luminosity_distance(redshifts, Omega_m0)
    min_mod_absolute_magnitude = 15.0
    max_mod_absolute_magnitude = 17.0
    margin_L = quad(l, min_mod_absolute_magnitude, max_mod_absolute_magnitude, args=(D_L, magnitudes, error_magnitudes, L0))[0]
    if np.isnan(margin_L) and zero_NaNs:
        margin_L = 0.0
    return margin_L


@timeit("Compute normalization factor") 
def normalization_factor(redshifts, magnitudes, error_magnitudes, guess=1.0, bounds_Omega_m0=(0.0, 2.5)): 
    L = lambda Omega_m0: marginalized_likelihood(Omega_m0, redshifts, magnitudes, error_magnitudes, L0=guess, zero_NaNs=True) 
    L0 = guess / quad(L, *bounds_Omega_m0)[0]                                                   
    return L0 


def find_best_fit_values(LIST_marginalized_likelihood):
    return np.nanargmax(LIST_marginalized_likelihood)


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



    # === Computation of marginalized likelihood, finding best values for Omega_m0 and Omega_Lambda ===
    # =================================================================================================

    # --- define variables ---
    LIST_Omega_m0 = np.linspace(0.0, 1.0, 400)

    # --- compute normalization factor for marginalized likelihood ---
    # L0 = normalization_factor(redshifts, magnitudes, error_magnitudes, guess=1e+124)
    L0 = 1.388170e+125

    # --- compute marginalized likelihood for every value in LIST_Omega_m0 and LIST_Omega_Lambda0 ---
    LIST_marginalized_likelihood = [marginalized_likelihood(Omega_m0, redshifts, magnitudes, error_magnitudes, L0) for Omega_m0 in LIST_Omega_m0]

    # --- find index of the value for Omega_m0 where the marginalized likelihood has its maximum ---
    Omega_m0_index = find_best_fit_values(LIST_marginalized_likelihood)
    Omega_m0_best = LIST_Omega_m0[Omega_m0_index]
    Omega_Lambda0_best = 1.0 - Omega_m0_best
    # marginalized_likelihood_best = LIST_marginalized_likelihood[Omega_m0_index]

    # --- print best Omega_m0_best and Omega_Lambda0_best ---
    print("==================================")
    print("Cosmological Parameter Estimation: ")
    print("----------------------------------")
    print(f"Omega_m0      = {Omega_m0_best:.3f}")
    print(f"Omega_Lambda0 = {Omega_Lambda0_best:.3f}")
    print("==================================")
    # =================================================================================================


    END_TOTAL_TIME = time.perf_counter()
    print(f"********** TOTAL COMPUTATION TIME: {END_TOTAL_TIME - START_TOTAL_TIME:.2f} seconds **********")


    # === Plot Omega_m0 vs. marginalized likelihood ===
    # =================================================
 
    # --- plot ---
    fig, ax = plt.subplots()

    plt.plot(LIST_Omega_m0, LIST_marginalized_likelihood, color='tab:blue')
    # plt.plot(Omega_m0_best, marginalized_likelihood_best, 'o', color='tab:red')
    plt.xlabel(r'$\Omega_{\text{m},0}$')
    plt.ylabel(r'$L_{\text{M}}(\Omega_{\text{m},0} \vert D)$')
    # plt.suptitle(r'$\texttt{MWE-marginalized-likelihood.py}$', fontsize=20)

    at = AnchoredText(fr'$\vb*{{\theta_{{\text{{best}}}}}} = ({Omega_m0_best:.2f}, {Omega_Lambda0_best:.2f})$', loc='upper right', borderpad=0.5)
    at.patch.set(boxstyle='round,pad=0.2', fc='w', ec='0.5', alpha=0.9)
    ax.add_artist(at)

    plt.grid(True)
    # plt.show()

    # --- save fig ---
    fig.savefig('../thesis/figures/plots/EPS/MWE-marginalized-likelihood.eps', format='eps', bbox_inches='tight')
    fig.savefig('../thesis/figures/plots/PNG/MWE-marginalized-likelihood.png', format='png', bbox_inches='tight', dpi = 250)
    fig.savefig('../thesis/figures/plots/PDF/MWE-marginalized-likelihood.pdf', format='pdf', bbox_inches='tight')
    # tikzplotlib.save('../thesis/figures/tikz/MWE-marginalized-likelihood.tex')

    # =================================================


if __name__ == "__main__":
    main()
