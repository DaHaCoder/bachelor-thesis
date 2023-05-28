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
from scipy import optimize as opt                                  #   for optimization and fit -- https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
from scipy import stats                                            #   for chi2 -- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html#scipy.stats.chi2

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
def mod_friedmann(E, z, Omega_m0, alpha):
    # expansion function E = H(z)/H_0
    return E * E - (1.0 - Omega_m0) * np.power(E, alpha) - Omega_m0 * np.power(1.0 + z, 3)


@njit
def deriv_mod_friedmann(E, _, Omega_m0, alpha):
    return 2.0 * E - alpha * (1.0 - Omega_m0) * np.power(E, alpha - 1.0)


def sol_friedmann(z, Omega_m0, alpha, friedmann, deriv_friedmann):
    # Solves the friedmann equation f(z) = 0 for z with derivative deriv_friedmann
    return opt.root(friedmann, 1.0, args=(z, Omega_m0, alpha), jac=deriv_friedmann).x[0]


@njit
def interp_integrand(z, sample_redshifts, sample_E):
    if z == 0.0:
        return 1.0

    E = np.interp(z, sample_redshifts, sample_E)
    return 1.0/E


def interp_integral(z, sample_redshifts, sample_E):
    # d_C/d_H = Integrate[1/E(z'), {z', 0, z}]
    return quad(interp_integrand, 0.0, z, args=(sample_redshifts, sample_E))[0]


def DGP_mod_luminosity_distance(z, Omega_m0, alpha):
    # Cosmological Parameters
    # =======================
    c = 299792.458           # speed of light in vacuum in km/s
    # h = 0.6766               # Planck Collaboration 2018, Table 7, Planck+BAO -- https://www.aanda.org/articles/aa/full_html/2020/09/aa33880-18/T7.html
    # H_0 = h*100.0            # hubble constant in km/s per Mpc
    H_0 = 1.0                # dependence on hubble constant is set into the mod_absolute_magnitude, see theoretical_magnitude
    d_H = c/H_0              # hubble distance
    # =======================

    sample_redshifts = np.linspace(0.0, max(z), 1000)
    sample_E = np.array([sol_friedmann(zi, Omega_m0, alpha, mod_friedmann, deriv_mod_friedmann) for zi in sample_redshifts])

    I = np.array([interp_integral(zi, sample_redshifts, sample_E) for zi in z])

    return (1.0 + z) * d_H * I


@njit
def theoretical_magnitude(mod_absolute_magnitude, mod_luminosity_distance):
    # mod_luminosity_distance := H_0 * luminosity_distance
    # mod_absolute_magnitude := old_absolute_magnitude - 5.0 * np.log10(H_0) + 25.0
    return mod_absolute_magnitude + 5.0 * np.log10(mod_luminosity_distance)
# ====================== #


@njit
def chi_square_analytic(magnitudes, error_magnitudes, theoretical_magnitudes):
    c1 = 0.0
    b0 = 0.0
    b1 = 0.0
    for m, sigma, m_th in zip(magnitudes, error_magnitudes, theoretical_magnitudes):
        c1 += 1.0/(sigma * sigma)
        b0 += (m_th - m)/(sigma * sigma)
        b1 += ((m_th - m)/sigma) * ((m_th - m)/sigma)
    return b1 - b0 * b0/c1


def likelihood(Omega_m0, alpha, redshifts, magnitudes, error_magnitudes, L0, zero_NaNs=False):
    mod_absolute_magnitude = 0.0
    D_L = DGP_mod_luminosity_distance(redshifts, Omega_m0, alpha)
    m_th = theoretical_magnitude(mod_absolute_magnitude, D_L)
    chi_2 = chi_square_analytic(magnitudes, error_magnitudes, m_th)
    L = L0 * np.exp(-0.5 * chi_2)
    if np.isnan(L) and zero_NaNs:
        L = 0.0
    return L


def likelihood_par_helper(i, j, Omega_m0, alpha, redshifts, magnitudes, error_magnitudes, L0):
    return i, j, likelihood(Omega_m0[i], alpha[j], redshifts, magnitudes, error_magnitudes, L0)


@timeit("Compute normalization factor")
def normalization_factor(redshifts, magnitudes, error_magnitudes, guess=1.0, bounds_m0=(0.0, 2.5), bounds_alpha=(0.0, 2.0)):
    L = lambda Omega_m0, alpha: likelihood(Omega_m0, alpha, redshifts, magnitudes, error_magnitudes, L0=guess, zero_NaNs=True)
    L0 = guess / dblquad(L, *bounds_alpha, *bounds_m0)[0]
    return L0


@timeit("Compute likelihood on grid")
def MATRIX_likelihood(Omega_m0, alpha, redshifts, magnitudes, error_magnitudes, L0):
    rows = len(Omega_m0)
    cols = len(alpha)

    # define matrix where 'rows' is the amount of rows and 'cols' the amount of columns
    MATRIX = np.zeros((rows, cols))
    likelihood_args = ((i, j, Omega_m0, alpha, redshifts, magnitudes, error_magnitudes, L0) for i, j in product(range(rows), range(cols)))

    with Pool() as pool:
        for i, j, L in pool.starmap(likelihood_par_helper, likelihood_args):
            MATRIX[i, j] = L

    # for args in likelihood_args:
    #     i, j, L = likelihood_par_helper(*args)
    #     MATRIX[i, j] = L

    return MATRIX.T


def find_best_fit_values(Omega_m0, alpha, MATRIX_likelihood):
    return np.unravel_index(np.nanargmax(MATRIX_likelihood), MATRIX_likelihood.shape)


def main():
    START_TOTAL_TIME = time.perf_counter()

    # Import data
    DATA_DIR = '../data/SN-data.txt'
    names, redshifts, magnitudes, error_magnitudes = np.loadtxt(DATA_DIR,
                                                                comments='#',
                                                                usecols=(0, 1, 2, 3),
                                                                dtype=np.dtype([('name', str),
                                                                                ('redshift', float),
                                                                                ('magnitude', float),
                                                                                ('sigma', float)]),
                                                                unpack=True)



    # === Computation of likelihood, finding best values for Omega_m0 and Omega_Lambda ===
    # ====================================================================================

    # --- define variables ---
    Omega_m0 = np.linspace(0.0, 1.0, 50)
    alpha    = np.linspace(0.0, 1.0, 50)
    
    # --- compute normalization factor ---
    # L0 = normalization_factor(redshifts, magnitudes, error_magnitudes, guess=1e+124)
    L0 = 1.0e+124

    # --- compute likelihood for every value in Omega_m0 and alpha ---
    MATRIX_like = MATRIX_likelihood(Omega_m0, alpha, redshifts, magnitudes, error_magnitudes, L0)
    
    # print()
    # print("==========================")
    # print(f"{L0 = :.6e}")
    # print(f"Sanity check: Riemann integral of likelihoods = {np.nansum(MATRIX_like) * np.diff(Omega_m0)[0] * np.diff(alpha)[0]:.6f}")
    # print()
    
    # --- find index of the values for Omega_m0 and alpha where MATRIX_like has its maximum ---
    alpha_index, Omega_m0_index = find_best_fit_values(Omega_m0, alpha, MATRIX_like)
    Omega_m0_best = Omega_m0[Omega_m0_index]
    alpha_best = alpha[alpha_index]

    # --- likelihood at its maximum ---
    max_MATRIX_like = MATRIX_like[alpha_index, Omega_m0_index]

    # --- compute MATRIX_like_summed_Omega_X0 ---
    MATRIX_like_summed_Omega_m0 = np.nansum(MATRIX_like, axis=0)
    MATRIX_like_summed_alpha = np.nansum(MATRIX_like, axis=1)
    # COMMENT FOR sum_X0 = np.nansum(MATRIX_like, axis=0)
    # for i in range(len(Omega_X0)):
    #    s = 0.0
    #    for j in range(len(Omega_Y0)):
    #        s += MATRIX_like[i, j]
    #    sum_X0[i] = s

    # --- print best Omega_m0_best and alpha_best ---
    print("================================")
    print("Cosmological Parameter Estimation:")
    print("----------------------------------")
    print(f"Omega_m0 = {Omega_m0_best:.3f}")
    print(f"alpha    = {alpha_best:.3f}")
    print("================================")
    # ====================================================================================
    

    END_TOTAL_TIME = time.perf_counter()
    print(f"********** TOTAL COMPUTATION TIME: {END_TOTAL_TIME - START_TOTAL_TIME:.2f} seconds **********")
   
    
    # === Plot Omega_X0 vs. likelihood for summed Omega_Y0 ===
    # ========================================================

    # --- plot Omega_m0 vs. likelihood for summed alpha ---
    fig = plt.figure()

    plt.plot(Omega_m0, MATRIX_like_summed_Omega_m0, label='data')
    plt.xlabel(r'$\Omega_{\text{m},0}$')
    plt.ylabel(r'$L_{\sum \alpha}(\Omega_{\text{m},0})$')
    plt.suptitle(r'$\\texttt{DGP_analytic_likelihood.py}$', fontsize=20)
    plt.grid(True)
    plt.show()

    # --- save fig ---
    fig.savefig('../thesis/figures/plots/EPS/DGP-analytic-likelihood_Omega-m0-vs-likelihood-summed-alpha.eps', format = 'eps', bbox_inches = 'tight')
    fig.savefig('../thesis/figures/plots/PNG/DGP-analytic-likelihood_Omega-m0-vs-likelihood-summed-alpha.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    fig.savefig('../thesis/figures/plots/PDF/DGP-analytic-likelihood_Omega-m0-vs-likelihood-summed-alpha.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/DGP-analytic-likelihood_Omega-m0-vs-likelihood-summed-alpha.tex')

    # --- plot alpha vs. likelihood for summed Omega_m0 ---
    fig = plt.figure()
    
    plt.plot(alpha, MATRIX_like_summed_alpha, label='data')
    plt.plot(alpha, MATRIX_like_summed_alpha_gauss_fit, linestyle='--', color='tab:orange', label='fit')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$L_{\sum \Omega_{\text{m},0}}(\alpha)$')
    plt.suptitle(r'$\texttt{DGP_analytic_likelihood.py}$', fontsize=20)
    plt.title(fr'best fit values: $(\Omega_{{\text{{m}},0}}, \alpha) = ({Omega_m0_best:.2f}, {alpha_best:.2f})$')
    plt.grid(True)
    plt.show()

    # --- save fig ---
    fig.savefig('../thesis/figures/plots/EPS/DGP-analytic-likelihood_alpha-vs-likelihood-summed-Omega-m0.eps', format = 'eps', bbox_inches = 'tight')
    fig.savefig('../thesis/figures/plots/PNG/DGP-analytic-likelihood_alpha-vs-likelihood-summed-Omega-m0.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    fig.savefig('../thesis/figures/plots/PDF/DGP-analytic-likelihood_alpha-vs-likelihood-summed-Omega-m0.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/DGP-analytic-likelihood_alpha-vs-likelihood-summed-Omega-m0.tex')
    # ========================================================


    # === Plot Omega_m0 vs. alpha vs. likelihood ===
    # ==============================================

    # --- compute likelihood ---
    Z = np.array(MATRIX_like)
    X, Y = np.meshgrid(Omega_m0, alpha)

    conf_int = [stats.chi2.cdf(s**2.0, 1) for s in range(1,5)]
    lvls = [max_MATRIX_like * np.exp(-0.5 * stats.chi2.ppf(ci, 2)) for ci in conf_int]
    lvls.sort()
    lvl_labels = [f"${k}\sigma$".format(k) for k in reversed(range(1,5))]

    # --- plot 3D ---
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_wireframe(X, Y, Z, edgecolor='tab:blue', alpha=0.3)
    ax.contourf(X, Y, Z, zdir='z', levels=lvls, offset=0, cmap='coolwarm')
    ax.scatter(Omega_m0_best, Omega_Lambda0_best, 0.0, 'o', color='red')
    ax.set(xlabel=r'$\Omega_{\text{m},0}$', ylabel=r'$\alpha$', zlabel=r'$L(\Omega_{\text{m},0}, \alpha)$')
    plt.suptitle(r'$\texttt{DGP_analytic_likelihood.py}$', fontsize=20)
    plt.title(fr'best fit values: $(\Omega_{{\text{{m}},0}}, \alpha) = ({Omega_m0_best:.2f}, {alpha_best:.2f})$')
    plt.grid(True)
    plt.show()

    # --- save fig ---
    fig.savefig('../thesis/figures/plots/EPS/DGP-analytic-likelihood_Omega-m0-vs-alpha-vs-likelihood.eps', format = 'eps', bbox_inches = 'tight')
    fig.savefig('../thesis/figures/plots/PNG/DGP-analytic-likelihood_Omega-m0-vs-alpha-vs-likelihood.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    fig.savefig('../thesis/figures/plots/PDF/DGP-analytic-likelihood_Omega-m0-vs-alpha-vs-likelihood.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/DGP-analytic-likelihood_Omega-m0-vs-alpha-vs-likelihood.tex')


    # --- plot 2D contour ---
    fig, ax = plt.subplots()

    CP = ax.contour(X, Y, Z, levels=lvls, cmap='coolwarm')
    plt.plot(Omega_m0_best, Omega_Lambda0_best, 'o', color='blue')
    fmt = {}
    for l, s in zip(CP.levels, lvl_labels):
        fmt[l] = s
    ax.clabel(CP, inline=True, fmt=fmt)

    plt.xlabel(r'$\Omega_{\text{m},0}$')
    plt.ylabel(r'$\alpha$')
    plt.suptitle(r'$\texttt{DGP_analytic_likelihood.py}$', fontsize=20)
    plt.title(fr'best fit values: $(\Omega_{{\text{{m}},0}}, \alpha) = ({Omega_m0_best:.2f}, {alpha_best:.2f})$')
    plt.grid(True)
    plt.show()

    # --- save fig ---
    fig.savefig('../thesis/figures/plots/EPS/DGP-analytic-likelihood_Omega-m0-vs-alpha.eps', format = 'eps', bbox_inches = 'tight')
    fig.savefig('../thesis/figures/plots/PNG/DGP-analytic-likelihood_Omega-m0-vs-alpha.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    fig.savefig('../thesis/figures/plots/PDF/DGP-analytic-likelihood_Omega-m0-vs-alpha.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/DGP-analytic-likelihood_Omega-m0-vs-alpha.tex')
    # ==============================================


if __name__ == "__main__":
    main()
