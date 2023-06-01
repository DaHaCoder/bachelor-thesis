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
def mod_friedmann(E, z, Omega_m0, alpha):
    return E * E - (1.0 - Omega_m0) * np.power(E, alpha) - Omega_m0 * np.power(1.0 + z, 3)


@njit
def deriv_mod_friedmann(E, _, Omega_m0, alpha):
    return 2.0 * E - alpha * (1.0 - Omega_m0) * np.power(E, alpha - 1.0)


def sol_friedmann(z, Omega_m0, alpha, mod_friedmann, mod_deriv_friedmann):
    # Solves the modified friedmann equation f(z) = 0 for z with exterior derivative mod_deriv_friedmann
    return opt.root(mod_friedmann, 1.0, args=(z, Omega_m0, alpha), jac=mod_deriv_friedmann).x[0]


@njit
def interp_integrand(z, sample_redshifts, sample_E):
    if z == 0.0:
        return 1.0

    E = np.interp(z, sample_redshifts, sample_E)
    return 1.0/E


def interp_integral(z, sample_redshifts, sample_E):
    # d_C/d_H = Integrate[1/E(z'), {z', 0, z}]
    return quad(interp_integrand, 0.0, z, args=(sample_redshifts, sample_E))[0]


def mod_luminosity_distance(z, Omega_m0, alpha):
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
def analytic_chi_square(magnitudes, error_magnitudes, theoretical_magnitudes):
    c1 = 0.0
    b0 = 0.0
    b1 = 0.0
    for m, sigma, m_th in zip(magnitudes, error_magnitudes, theoretical_magnitudes):
        c1 += 1.0/(sigma * sigma)
        b0 += (m_th - m)/(sigma * sigma)
        b1 += ((m_th - m)/sigma) * ((m_th - m)/sigma)
    return b1 - b0 * b0/c1


def chi2(Omega_m0, alpha, redshifts, magnitudes, error_magnitudes, zero_NaNs=False):
    mod_absolute_magnitude = 0.0
    D_L = mod_luminosity_distance(redshifts, Omega_m0, alpha)
    m_th = theoretical_magnitude(mod_absolute_magnitude, D_L)
    chi_2 = analytic_chi_square(magnitudes, error_magnitudes, m_th)
    if np.isnan(chi_2) and zero_NaNs:
        chi_2 = 0.0
    return chi_2 


def chi2_par_helper(i, j, Omega_m0, alpha, redshifts, magnitudes, error_magnitudes):
    return i, j, chi2(Omega_m0[i], alpha[j], redshifts, magnitudes, error_magnitudes)


@timeit("Compute chi2 on grid")
def MATRIX_chi2(Omega_m0, alpha, redshifts, magnitudes, error_magnitudes):
    rows = len(Omega_m0)
    cols = len(alpha)

    # define matrix where 'rows' is the amount of rows and 'cols' the amount of columns
    MATRIX = np.zeros((rows, cols))

    chi2_args = ((i, j, Omega_m0, alpha, redshifts, magnitudes, error_magnitudes) for i, j in product(range(rows), range(cols)))

    with Pool() as pool:
        for i, j, chi_2 in pool.starmap(chi2_par_helper, chi2_args):
            MATRIX[i, j] = chi_2

    # for args in chi2_args:
    #     i, j, L = chi2_par_helper(*args)
    #     MATRIX[i, j] = L

    return MATRIX.T


def find_best_fit_values(Omega_m0, alpha, MATRIX_chi2):
    return np.unravel_index(np.nanargmin(MATRIX_chi2), MATRIX_chi2.shape)


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



    # === Computation of MATRIX_chi_square, finding best values for Omega_m0 and alpha ===
    # ====================================================================================

    # --- define variables ---
    Omega_m0 = np.linspace(0.0, 0.6, 400)
    alpha    = np.linspace(-11.0, 2.0, 400)
    
    # --- compute MATRIX_chi_square for every value in Omega_m0 and alpha ---
    MATRIX_chi_square = MATRIX_chi2(Omega_m0, alpha, redshifts, magnitudes, error_magnitudes)
    
    # --- find index of the values for Omega_m0 and alpha where MATRIX_chi_square has its minimum ---
    alpha_index, Omega_m0_index = find_best_fit_values(Omega_m0, alpha, MATRIX_chi_square)
    Omega_m0_best = Omega_m0[Omega_m0_index]
    alpha_best = alpha[alpha_index]
    
    min_MATRIX_chi_square = MATRIX_chi_square[alpha_index, Omega_m0_index]

    # --- compute MATRIX_chi_square_summed_Omega_m0_X0 ---
    MATRIX_chi_square_summed_Omega_m0 = np.nansum(MATRIX_chi_square, axis=0)
    MATRIX_chi_square_summed_alpha = np.nansum(MATRIX_chi_square, axis=1)

    # --- print best Omega_m0_best and alpha_best ---
    print("==================================")
    print("Cosmological Parameter Estimation:")
    print("----------------------------------")
    print(f"Omega_m0 = {Omega_m0_best:.3f}")
    print(f"alpha    = {alpha_best:.3f}")
    print("==================================")
    # ====================================================================================
    

    END_TOTAL_TIME = time.perf_counter()
    print(f"********** TOTAL COMPUTATION TIME: {END_TOTAL_TIME - START_TOTAL_TIME:.2f} seconds **********")


    # === Plot Omega_m0 vs. MATRIX_chi_square for summed alpha and vice versa ===
    # ===========================================================================

    # --- plot Omega_m0 vs. MATRIX_chi_square for summed alpha ---
    fig, ax = plt.subplots()

    plt.plot(Omega_m0, MATRIX_chi_square_summed_Omega_m0, label='data')
    # plt.plot(Omega_m0_best, min_MATRIX_chi_square, 'o', color='red')
    plt.xlabel(r'$\Omega_{\text{m},0}$', fontsize=16)
    plt.ylabel(r'$\chi_{\text{A}, \sum \alpha}^2(\Omega_{\text{m},0} \vert D)$', fontsize=16)
    # plt.suptitle(r'$\texttt{DGP-analytic-chi2.py}$', fontsize=20)
    ax.tick_params(labelsize=14)
    plt.grid(True)
    
    # at = AnchoredText(fr'$(\Omega_{{\text{{m}}, 0, \text{{best}}}}, \alpha_{{\text{{best}}}}) = ({Omega_m0_best:.2f}, {alpha_best:.2f})$', loc='upper right', borderpad=0.5, prop=dict(fontsize=16))
    at = AnchoredText(fr'$\vb*{{\theta_{{\text{{best}}}}}} = ({Omega_m0_best:.2f}, {alpha_best:.2f})$', loc='upper right', borderpad=0.5, prop=dict(fontsize=16))
    at.patch.set(boxstyle='round,pad=0.2', fc='w', ec='0.5', alpha=0.9)
    ax.add_artist(at)
    
    # plt.show()

    # --- save fig ---
    fig.savefig('../thesis/figures/plots/EPS/DGP-analytic-chi2_Omega-m0-vs-chi2-summed-alpha.eps', format='eps', bbox_inches='tight')
    fig.savefig('../thesis/figures/plots/PNG/DGP-analytic-chi2_Omega-m0-vs-chi2-summed-alpha.png', format='png', bbox_inches='tight', dpi=250)
    fig.savefig('../thesis/figures/plots/PDF/DGP-analytic-chi2_Omega-m0-vs-chi2-summed-alpha.pdf', format='pdf', bbox_inches='tight')
    # tikzplotlib.save('../thesis/figures/tikz/DGP-analytic-chi2_Omega-m0-vs-chi2-summed-alpha.tex')

    # --- plot alpha vs. MATRIX_chi_square for summed Omega_m0 ---
    fig, ax = plt.subplots()

    plt.plot(alpha, MATRIX_chi_square_summed_alpha, label='data')
    # plt.plot(alpha_best, min_MATRIX_chi_square, 'o', color='red')
    plt.xlabel(r'$\alpha$', fontsize=16)
    plt.ylabel(r'$\chi_{\text{A}, \sum \Omega_{\text{m}, 0}}^2(\alpha \vert D)$', fontsize=16)
    # plt.suptitle(r'$\texttt{DGP-analytic-chi2.py}$', fontsize=20)
    ax.tick_params(labelsize=14)
    plt.grid(True)
    
    # at = AnchoredText(fr'$(\Omega_{{\text{{m}}, 0, \text{{best}}}}, \alpha_{{\text{{best}}}}) = ({Omega_m0_best:.2f}, {alpha_best:.2f})$', loc='upper right', borderpad=0.5, prop=dict(fontsize=16))
    at = AnchoredText(fr'$\vb*{{\theta_{{\text{{best}}}}}} = ({Omega_m0_best:.2f}, {alpha_best:.2f})$', loc='upper right', borderpad=0.5, prop=dict(fontsize=16))
    at.patch.set(boxstyle='round,pad=0.2', fc='w', ec='0.5', alpha=0.9)
    ax.add_artist(at)
    
    # plt.show()

    # --- save fig ---
    fig.savefig('../thesis/figures/plots/EPS/DGP-analytic-chi2_alpha-vs-chi2-summed-Omega-m0.eps', format='eps', bbox_inches='tight')
    fig.savefig('../thesis/figures/plots/PNG/DGP-analytic-chi2_alpha-vs-chi2-summed-Omega-m0.png', format='png', bbox_inches='tight', dpi=250)
    fig.savefig('../thesis/figures/plots/PDF/DGP-analytic-chi2_alpha-vs-chi2-summed-Omega-m0.pdf', format='pdf', bbox_inches='tight')
    # tikzplotlib.save('../thesis/figures/tikz/DGP-analytic-chi2_alpha-vs-chi2-summed-Omega-m0.tex')
    # ===========================================================================


    # === Plot Omega_m0 vs. MATRIX_chi_square at alpha_best and vice versa ===
    # ========================================================================

    # --- plot Omega_m0 vs. MATRIX_chi_square at alpha_best ---
    fig, ax = plt.subplots()

    plt.plot(Omega_m0, MATRIX_chi_square[alpha_index, :], label='data')
    plt.plot(Omega_m0_best, min_MATRIX_chi_square, 'o', color='red')
    plt.xlabel(r'$\Omega_{\text{m}, 0}$', fontsize=16)
    plt.ylabel(r'$\chi_{\text{A}}^2(\Omega_{\text{m}, 0}, \alpha_{\text{best}} \vert D)$', fontsize=16)
    # plt.suptitle(r'$\texttt{DGP-analytic-chi2.py}$', fontsize=20)
    ax.tick_params(labelsize=14)
    plt.grid(True)
    
    # at = AnchoredText(fr'$(\Omega_{{\text{{m}}, 0, \text{{best}}}}, \alpha_{{\text{{best}}}}) = ({Omega_m0_best:.2f}, {alpha_best:.2f})$', loc='upper right', borderpad=0.5, prop=dict(fontsize=16))
    at = AnchoredText(fr'$\vb*{{\theta_{{\text{{best}}}}}} = ({Omega_m0_best:.2f}, {alpha_best:.2f})$', loc='upper right', borderpad=0.5, prop=dict(fontsize=16))
    at.patch.set(boxstyle='round,pad=0.2', fc='w', ec='0.5', alpha=0.9)
    ax.add_artist(at)
    
    # plt.show()

    # --- save fig ---
    fig.savefig('../thesis/figures/plots/EPS/DGP-analytic-chi2_Omega-m0-vs-chi2-at-alpha-best.eps', format='eps', bbox_inches='tight')
    fig.savefig('../thesis/figures/plots/PNG/DGP-analytic-chi2_Omega-m0-vs-chi2-at-alpha-best.png', format='png', bbox_inches='tight', dpi=250)
    fig.savefig('../thesis/figures/plots/PDF/DGP-analytic-chi2_Omega-m0-vs-chi2-at-alpha-best.pdf', format='pdf', bbox_inches='tight')
    # tikzplotlib.save('../thesis/figures/tikz/DGP-analytic-chi2_Omega-m0-vs-chi2-at-alpha-best.tex')

    # --- plot alpha vs. chi2 at Omega_m0_best ---
    fig, ax = plt.subplots()

    plt.plot(alpha, MATRIX_chi_square[:, Omega_m0_index], label='data')
    plt.plot(alpha_best, min_MATRIX_chi_square, 'o', color='red')
    plt.xlabel(r'$\alpha$', fontsize=16)
    plt.ylabel(r'$\chi_{\text{A}}^2(\Omega_{\text{m}, 0, \text{best}}, \alpha \vert D)$', fontsize=16)
    # plt.suptitle(r'$\texttt{DGP_analytic_chi2.py}$', fontsize=20)
    ax.tick_params(labelsize=14)
    plt.grid(True)
    
    # at = AnchoredText(fr'$(\Omega_{{\text{{m}}, 0, \text{{best}}}}, \alpha_{{\text{{best}}}}) = ({Omega_m0_best:.2f}, {alpha_best:.2f})$', loc='upper right', borderpad=0.5, prop=dict(fontsize=16))
    at = AnchoredText(fr'$\vb*{{\theta_{{\text{{best}}}}}} = ({Omega_m0_best:.2f}, {alpha_best:.2f})$', loc='upper right', borderpad=0.5, prop=dict(fontsize=16))
    at.patch.set(boxstyle='round,pad=0.2', fc='w', ec='0.5', alpha=0.9)
    ax.add_artist(at)
    
    # plt.show()

    # --- save fig ---
    fig.savefig('../thesis/figures/plots/EPS/DGP-analytic-chi2_alpha-vs-chi2-at-Omega-m0-best.eps', format='eps', bbox_inches='tight')
    fig.savefig('../thesis/figures/plots/PNG/DGP-analytic-chi2_alpha-vs-chi2-at-Omega-m0-best.png', format='png', bbox_inches='tight', dpi=250)
    fig.savefig('../thesis/figures/plots/PDF/DGP-analytic-chi2_alpha-vs-chi2-at-Omega-m0-best.pdf', format='pdf', bbox_inches='tight')
    # tikzplotlib.save('../thesis/figures/tikz/DGP-analytic-chi2_alpha-vs-chi2-at-Omega-m0-best.tex')
    # =======================================================================


    # === Plot Omega_m0 vs. alpha vs. MATRIX_chi_square ===
    # =====================================================

    # --- compute chi2 ---
    Z = np.array(MATRIX_chi_square)
    X, Y = np.meshgrid(Omega_m0, alpha)

    conf_int = [stats.chi2.cdf(s**2.0, 1) for s in range(1,5)]
    lvls = [stats.chi2.ppf(ci, 2) + min_MATRIX_chi_square for ci in conf_int]
    lvl_labels = [f"${k}\sigma$".format(k) for k in range(1,5)]

    # --- plot 3D ---
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_wireframe(X, Y, Z, edgecolor='tab:blue', alpha=0.5, linewidths=1.0)
    ax.contour(X, Y, Z, zdir='z', levels=lvls, cmap='coolwarm_r', linewidths=1.0)
    ax.scatter(Omega_m0_best, alpha_best, min_MATRIX_chi_square, '.', color='red', s=1.0)
    ax.set_xlabel(xlabel=r'$\Omega_{\text{m},0}$', fontsize=12)
    ax.set_ylabel(ylabel=r'$\alpha$', fontsize=12)
    ax.set_zlabel(zlabel=r'$\chi_{\text{A}}^2(\Omega_{\text{m},0}, \alpha \vert D)$', fontsize=12)
    ax.tick_params(axis='both', width=10, labelsize=8, pad=0)
    # plt.suptitle(r'$\texttt{DGP-analytic-chi2.py}$', fontsize=20)
    plt.grid(True)
    # plt.show()

    # --- save fig ---
    fig.savefig('../thesis/figures/plots/EPS/DGP-analytic-chi2_Omega-m0-vs-alpha-vs-chi2.eps', format='eps', bbox_inches='tight')
    fig.savefig('../thesis/figures/plots/PNG/DGP-analytic-chi2_Omega-m0-vs-alpha-vs-chi2.png', format='png', bbox_inches='tight', dpi=250)
    fig.savefig('../thesis/figures/plots/PDF/DGP-analytic-chi2_Omega-m0-vs-alpha-vs-chi2.pdf', format='pdf', bbox_inches='tight')
    # tikzplotlib.save('../thesis/figures/tikz/DGP-analytic-chi2_Omega-m0-vs-alpha-vs-chi2.tex')


    # --- plot 2D contour ---
    fig, ax = plt.subplots()

    CP = ax.contour(X, Y, Z, levels=lvls, cmap='coolwarm_r', linewidths=1.0)
    plt.plot(Omega_m0_best, alpha_best, '.', color='red')
    fmt = {}
    for l, s in zip(CP.levels, lvl_labels):
        fmt[l] = s
    ax.clabel(CP, inline=True, fmt=fmt)

    plt.xlabel(r'$\Omega_{\text{m}, 0}$', fontsize=14)
    plt.ylabel(r'$\alpha$', fontsize=14)
    # plt.suptitle(r'$\texttt{DGP-analytic-chi2.py}$', fontsize=20)
    ax.tick_params(labelsize=10)
    plt.grid(True)
    
    # at = AnchoredText(fr'$(\Omega_{{\text{{m}}, 0, \text{{best}}}}, \alpha_{{\text{{best}}}}) = ({Omega_m0_best:.2f}, {alpha_best:.2f})$', loc='upper right', borderpad=0.5, prop=dict(fontsize=16))
    at = AnchoredText(fr'$\vb*{{\theta_{{\text{{best}}}}}} = ({Omega_m0_best:.2f}, {alpha_best:.2f})$', loc='lower left', borderpad=0.5, prop=dict(fontsize=14))
    at.patch.set(boxstyle='round,pad=0.2', fc='w', ec='0.5', alpha=0.9)
    ax.add_artist(at)

    # plt.show()

    # --- save fig ---
    fig.savefig('../thesis/figures/plots/EPS/DGP-analytic-chi2_Omega-m0-vs-alpha-full.eps', format='eps', bbox_inches='tight')
    fig.savefig('../thesis/figures/plots/PNG/DGP-analytic-chi2_Omega-m0-vs-alpha-full.png', format='png', bbox_inches='tight', dpi=250)
    fig.savefig('../thesis/figures/plots/PDF/DGP-analytic-chi2_Omega-m0-vs-alpha-full.pdf', format='pdf', bbox_inches='tight')
    # tikzplotlib.save('../thesis/figures/tikz/DGP-analytic-chi2_Omega-m0-vs-alpha-full.tex')

    # --- plot 2D contour for alpha in [0.0, 2.0] ---
    fig, ax = plt.subplots()

    CP = ax.contour(X, Y, Z, levels=lvls, cmap='coolwarm_r', linewidths=1.0)
    plt.plot(Omega_m0_best, alpha_best, '.', color='red')
    fmt = {}
    for l, s in zip(CP.levels, lvl_labels):
        fmt[l] = s
    ax.clabel(CP, inline=True, fmt=fmt)

    plt.xlabel(r'$\Omega_{\text{m}, 0}$', fontsize=16)
    plt.ylabel(r'$\alpha$', fontsize=16)
    plt.ylim(0.0, 2.0)
    # plt.suptitle(r'$\texttt{DGP-analytic-chi2.py}$', fontsize=20)
    ax.tick_params(labelsize=14)
    plt.grid(True)
    
    # at = AnchoredText(fr'$(\Omega_{{\text{{m}}, 0, \text{{best}}}}, \alpha_{{\text{{best}}}}) = ({Omega_m0_best:.2f}, {alpha_best:.2f})$', loc='upper right', borderpad=0.5, prop=dict(fontsize=16))
    at = AnchoredText(fr'$\vb*{{\theta_{{\text{{best}}}}}} = ({Omega_m0_best:.2f}, {alpha_best:.2f})$', loc='upper right', borderpad=0.5, prop=dict(fontsize=16))
    at.patch.set(boxstyle='round,pad=0.2', fc='w', ec='0.5', alpha=0.9)
    ax.add_artist(at)

    # plt.show()

    # --- save fig ---
    fig.savefig('../thesis/figures/plots/EPS/DGP-analytic-chi2_Omega-m0-vs-alpha-0-2.eps', format='eps', bbox_inches='tight')
    fig.savefig('../thesis/figures/plots/PNG/DGP-analytic-chi2_Omega-m0-vs-alpha-0-2.png', format='png', bbox_inches='tight', dpi=250)
    fig.savefig('../thesis/figures/plots/PDF/DGP-analytic-chi2_Omega-m0-vs-alpha-0-2.pdf', format='pdf', bbox_inches='tight')
    # tikzplotlib.save('../thesis/figures/tikz/DGP-analytic-chi2_Omega-m0-vs-alpha-0-2.tex')
    # =====================================================


if __name__ == "__main__":
    main()
