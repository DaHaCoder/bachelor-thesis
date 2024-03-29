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
# plt.rcParams['font.size'] = 9
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
def expansion_function(z, Omega_r0, Omega_m0, Omega_Lambda0):
    Omega_K0 = 1.0 - Omega_r0 - Omega_m0 - Omega_Lambda0
    return np.sqrt(Omega_r0 * np.power(1.0 + z, 4) + Omega_m0 * np.power(1.0 + z, 3) + Omega_K0 * np.power(1.0 + z, 2) + Omega_Lambda0)


@njit
def integrand(z, Omega_r0, Omega_m0, Omega_Lambda0):
    if z == 0.0:
        return 1.0
    E = expansion_function(z, Omega_r0, Omega_m0, Omega_Lambda0)
    return 1.0/E


def integral(z, Omega_r0, Omega_m0, Omega_Lambda0):
    # d_C/d_H = Integrate[1/E(z'), {z',0,z}]
    return quad(integrand, 0.0, z, args=(Omega_r0, Omega_m0, Omega_Lambda0))[0]


def mod_luminosity_distance(z, Omega_m0, Omega_Lambda0):
    # Cosmological Parameters
    # =======================
    c = 299792.458           # speed of light in vacuum in km/s
    # h = 0.6766               # Planck Collaboration 2018, Table 7, Planck+BAO -- https://www.aanda.org/articles/aa/full_html/2020/09/aa33880-18/T7.html
    # H_0 = h*100.0            # hubble constant in km/s per Mpc
    H_0 = 1.0                # dependence on hubble constant is set into the mod_absolute_magnitude, see theoretical_magnitude
    d_H = c/H_0              # hubble distance
    Omega_r0 = 0.0           # assume no radiation
    # =======================

    I = np.array([integral(zi, Omega_r0, Omega_m0, Omega_Lambda0) for zi in z])

    Omega_K0 = 1.0 - Omega_r0 - Omega_m0 - Omega_Lambda0

    if Omega_K0 < 0.0:
        comoving_distance = d_H * 1.0/np.sqrt(abs(Omega_K0)) * np.sin(np.sqrt(abs(Omega_K0)) * I)

    elif Omega_K0 == 0.0:
        comoving_distance = d_H * I

    elif Omega_K0 > 0.0:
        comoving_distance = d_H * 1.0/np.sqrt(Omega_K0) * np.sinh(np.sqrt(Omega_K0) * I)

    return (1.0 + z) * comoving_distance


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


def chi2(Omega_m0, Omega_Lambda0, redshifts, magnitudes, error_magnitudes):
    mod_absolute_magnitude = 0.0
    D_L = mod_luminosity_distance(redshifts, Omega_m0, Omega_Lambda0)
    m_th = theoretical_magnitude(mod_absolute_magnitude, D_L)
    chi_2 = analytic_chi_square(magnitudes, error_magnitudes, m_th)
    return chi_2 


def chi2_par_helper(i, j, Omega_m0, Omega_Lambda0, redshifts, magnitudes, error_magnitudes):
    return i, j, chi2(Omega_m0[i], Omega_Lambda0[j], redshifts, magnitudes, error_magnitudes)


@timeit("Compute chi2 on grid")
def MATRIX_chi2(Omega_m0, Omega_Lambda0, redshifts, magnitudes, error_magnitudes):
    rows = len(Omega_m0)
    cols = len(Omega_Lambda0)

    # define matrix where 'rows' is the amount of rows and 'cols' the amount of columns
    MATRIX = np.zeros((rows, cols))

    chi2_args = ((i, j, Omega_m0, Omega_Lambda0, redshifts, magnitudes, error_magnitudes) for i, j in product(range(rows), range(cols)))

    with Pool() as pool:
        for i, j, chi_2 in pool.starmap(chi2_par_helper, chi2_args):
            MATRIX[i, j] = chi_2

    # for args in chi2_args:
    #     i, j, L = chi2_par_helper(*args)
    #     MATRIX[i, j] = L

    return MATRIX.T


def find_best_fit_values(Omega_m0, Omega_Lambda0, MATRIX_chi2):
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



    # === Computation of chi2, finding best values for Omega_m0 and Omega_Lambda0 ===
    # ===============================================================================

    # --- define variables ---
    Omega_m0      = np.linspace(0.0, 1.0, 400)
    Omega_Lambda0 = np.linspace(0.0, 1.4, 400)
    
    # --- compute chi2 for every value in Omega_m0 and Omega_Lambda0 ---
    MATRIX_chi_square = MATRIX_chi2(Omega_m0, Omega_Lambda0, redshifts, magnitudes, error_magnitudes)
    
    # --- find index of the values for Omega_m0 and Omega_Lambda0 where MATRIX_chi_square has its minimum ---
    Omega_Lambda0_index, Omega_m0_index = find_best_fit_values(Omega_m0, Omega_Lambda0, MATRIX_chi_square)
    Omega_m0_best = Omega_m0[Omega_m0_index]
    Omega_Lambda0_best = Omega_Lambda0[Omega_Lambda0_index]
    
    min_MATRIX_chi_square = MATRIX_chi_square[Omega_Lambda0_index, Omega_m0_index]

    # --- compute MATRIX_chi_square_summed_Omega_X0 ---
    MATRIX_chi_square_summed_Omega_m0 = np.nansum(MATRIX_chi_square, axis=1)
    MATRIX_chi_square_summed_Omega_Lambda0 = np.nansum(MATRIX_chi_square, axis=0)

    # --- print Omega_m0_best and Omega_Lambda0_best ---
    print("==================================")
    print("Cosmological Parameter Estimation:")
    print("----------------------------------")
    print(f"Omega_m0      = {Omega_m0_best:.3f}")
    print(f"Omega_Lambda0 = {Omega_Lambda0_best:.3f}")
    print("==================================")
    # ================================================================================
    

    END_TOTAL_TIME = time.perf_counter()
    print(f"********** TOTAL COMPUTATION TIME: {END_TOTAL_TIME - START_TOTAL_TIME:.2f} seconds **********")
    

    # === Plot Omega_X0 vs. chi2 for summed Omega_Y0 ===
    # ==================================================

    # --- plot Omega_m0 vs. chi2 for summed Omega_Lambda0 ---
    fig, ax = plt.subplots()

    plt.plot(Omega_m0, MATRIX_chi_square_summed_Omega_Lambda0, label='data')
    # plt.plot(Omega_m0_best, min_MATRIX_chi_square, 'o', color='red')
    plt.xlabel(r'$\Omega_{\text{m},0}$', fontsize=16)
    plt.ylabel(r'$\chi_{\text{A}, \sum \Omega_{\Lambda,0}}^2(\Omega_{\text{m},0} \vert D)$', fontsize=16)
    # plt.suptitle(r'$\texttt{Lambda-CDM-analytic-chi2.py}$', fontsize=20)
    plt.grid(True)
    
    at = AnchoredText(fr'$\vb*{{\theta}}_{{\text{{best}}}} = ({Omega_m0_best:.2f}, {Omega_Lambda0_best:.2f})$', loc='upper right', borderpad=0.5, prop=dict(fontsize=16))
    at.patch.set(boxstyle='round,pad=0.2', fc='w', ec='0.5', alpha=0.9)
    ax.add_artist(at)

    # plt.show()

    # --- save fig ---
    fig.savefig('../thesis/figures/plots/EPS/Lambda-CDM-analytic-chi2_Omega-m0-vs-chi2-summed-Omega-Lambda0.eps', format='eps', bbox_inches='tight')
    fig.savefig('../thesis/figures/plots/PNG/Lambda-CDM-analytic-chi2_Omega-m0-vs-chi2-summed-Omega-Lambda0.png', format='png', bbox_inches='tight', dpi=250)
    fig.savefig('../thesis/figures/plots/PDF/Lambda-CDM-analytic-chi2_Omega-m0-vs-chi2-summed-Omega-Lambda0.pdf', format='pdf', bbox_inches='tight')
    # tikzplotlib.save('../thesis/figures/tikz/Lambda-CDM-analytic-chi2_Omega-m0-vs-chi2-summed-Omega-Lambda0.tex')

    # --- plot Omega_Lambda0 vs. chi2 for summed Omega_m0 ---
    fig, ax = plt.subplots()
    
    plt.plot(Omega_Lambda0, MATRIX_chi_square_summed_Omega_m0, label='data')
    # plt.plot(Omega_Lambda0_best, min_MATRIX_chi_square, 'o', color='red')
    plt.xlabel(r'$\Omega_{\Lambda,0}$', fontsize=16)
    plt.ylabel(r'$\chi_{\text{A}, \sum \Omega_{\text{m},0}}^{2}(\Omega_{\Lambda,0} \vert D)$', fontsize=16)
    # plt.suptitle(r'$\texttt{Lambda-CDM-analytic-chi2.py}$', fontsize=20)
    plt.grid(True)
 
    at = AnchoredText(fr'$\vb*{{\theta}}_{{\text{{best}}}} = ({Omega_m0_best:.2f}, {Omega_Lambda0_best:.2f})$', loc='upper right', borderpad=0.5, prop=dict(fontsize=16))
    at.patch.set(boxstyle='round,pad=0.2', fc='w', ec='0.5', alpha=0.9)
    ax.add_artist(at)

    # plt.show()

    # --- save fig ---
    fig.savefig('../thesis/figures/plots/EPS/Lambda-CDM-analytic-chi2_Omega-Lambda0-vs-chi2-summed-Omega-m0.eps', format='eps', bbox_inches='tight')
    fig.savefig('../thesis/figures/plots/PNG/Lambda-CDM-analytic-chi2_Omega-Lambda0-vs-chi2-summed-Omega-m0.png', format='png', bbox_inches='tight', dpi=250)
    fig.savefig('../thesis/figures/plots/PDF/Lambda-CDM-analytic-chi2_Omega-Lambda0-vs-chi2-summed-Omega-m0.pdf', format='pdf', bbox_inches='tight')
    # tikzplotlib.save('../thesis/figures/tikz/Lambda-CDM-analytic-chi2_Omega-Lambda0-vs-chi2-summed-Omega-m0.tex')
    # ==================================================

   
    # === Plot Omega_X0 vs. chi2 at Omega_Y0_best ===
    # ===============================================

    # --- plot Omega_m0 vs. chi2 at Omega_Lambda0_best ---
    fig, ax = plt.subplots()

    plt.plot(Omega_m0, MATRIX_chi_square[Omega_Lambda0_index, :], label='data')
    plt.plot(Omega_m0_best, min_MATRIX_chi_square, 'o', color='red')
    plt.xlabel(r'$\Omega_{\text{m},0}$', fontsize=16)
    plt.ylabel(r'$\chi_{\text{A}}^2(\Omega_{\text{m},0}, \Omega_{\Lambda, 0, \text{best}} \vert D)$', fontsize=16)
    # plt.suptitle(r'$\texttt{Lambda-CDM-analytic-chi2.py}$', fontsize=20)
    ax.tick_params(labelsize=14)
    plt.grid(True)

    at = AnchoredText(fr'$\vb*{{\theta}}_{{\text{{best}}}} = ({Omega_m0_best:.2f}, {Omega_Lambda0_best:.2f})$', loc='upper right', borderpad=0.5, prop=dict(fontsize=16))
    at.patch.set(boxstyle='round,pad=0.2', fc='w', ec='0.5', alpha=0.9)
    ax.add_artist(at)

    # plt.show()

    # --- save fig ---
    fig.savefig('../thesis/figures/plots/EPS/Lambda-CDM-analytic-chi2_Omega-m0-vs-chi2-at-Omega-Lambda0-best.eps', format='eps', bbox_inches='tight')
    fig.savefig('../thesis/figures/plots/PNG/Lambda-CDM-analytic-chi2_Omega-m0-vs-chi2-at-Omega-Lambda0-best.png', format='png', bbox_inches='tight', dpi=250)
    fig.savefig('../thesis/figures/plots/PDF/Lambda-CDM-analytic-chi2_Omega-m0-vs-chi2-at-Omega-Lambda0-best.pdf', format='pdf', bbox_inches='tight')
    # tikzplotlib.save('../thesis/figures/tikz/Lambda-CDM-analytic-chi2_Omega-m0-vs-chi2-at-Omega-Lambda0-best.tex')

    # --- plot Omega_Lambda0 vs. chi2 at Omega_m0_best ---
    fig, ax = plt.subplots()

    plt.plot(Omega_Lambda0, MATRIX_chi_square[:, Omega_m0_index], label='data')
    plt.plot(Omega_Lambda0_best, min_MATRIX_chi_square, 'o', color='red')
    plt.xlabel(r'$\Omega_{\Lambda,0}$', fontsize=16)
    plt.ylabel(r'$\chi_{\text{A}}^2(\Omega_{\text{m}, 0, \text{best}}, \Omega_{\Lambda,0} \vert D)$', fontsize=16)
    # plt.suptitle(r'$\texttt{Lambda-CDM-analytic-chi2.py}$', fontsize=20)
    ax.tick_params(labelsize=14)
    plt.grid(True)

    at = AnchoredText(fr'$\vb*{{\theta}}_{{\text{{best}}}} = ({Omega_m0_best:.2f}, {Omega_Lambda0_best:.2f})$', loc='upper right', borderpad=0.5, prop=dict(fontsize=16))
    at.patch.set(boxstyle='round,pad=0.2', fc='w', ec='0.5', alpha=0.9)
    ax.add_artist(at)

    # plt.show()

    # --- save fig ---
    fig.savefig('../thesis/figures/plots/EPS/Lambda-CDM-analytic-chi2_Omega-Lambda0-vs-chi2-at-Omega-m0-best.eps', format='eps', bbox_inches='tight')
    fig.savefig('../thesis/figures/plots/PNG/Lambda-CDM-analytic-chi2_Omega-Lambda0-vs-chi2-at-Omega-m0-best.png', format='png', bbox_inches='tight', dpi=250)
    fig.savefig('../thesis/figures/plots/PDF/Lambda-CDM-analytic-chi2_Omega-Lambda0-vs-chi2-at-Omega-m0-best.pdf', format='pdf', bbox_inches='tight')
    # tikzplotlib.save('../thesis/figures/tikz/Lambda-CDM-analytic-chi2_Omega-Lambda0-vs-chi2-at-Omega-m0-best.tex')
    # ===============================================


    # === Plot Omega_m0 vs. Omega_Lambda0 vs. chi2 ===
    # ================================================

    # --- compute chi2 ---
    Z = np.array(MATRIX_chi_square)
    X, Y = np.meshgrid(Omega_m0, Omega_Lambda0)

    conf_int = [stats.chi2.cdf(s**2.0, 1) for s in range(1,5)]
    lvls = [stats.chi2.ppf(ci, 2) + min_MATRIX_chi_square for ci in conf_int]
    lvl_labels = [f'${k}\sigma$'.format(k) for k in range(1,5)]

    # --- plot 3D ---
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_wireframe(X, Y, Z, edgecolor='tab:blue', alpha=0.3, linewidths=1.0)
    ax.contour(X, Y, Z, zdir='z', levels=lvls, cmap='coolwarm_r', linewidths=1.0)
    ax.scatter(Omega_m0_best, Omega_Lambda0_best, min_MATRIX_chi_square, 'o', color='red', s=1.0)
    ax.set_xlabel(xlabel=r'$\Omega_{\text{m},0}$', fontsize=12)
    ax.set_ylabel(ylabel=r'$\Omega_{\Lambda,0}$', fontsize=12)
    ax.set_zlabel(zlabel=r'$\chi_{\text{A}}^2(\Omega_{\text{m},0}, \Omega_{\Lambda,0} \vert D)$', fontsize=12)
    ax.set_zlim(min_MATRIX_chi_square, 1499.0)
    ax.tick_params(axis='both', width=10, labelsize=6, pad=0)
    # plt.suptitle(r'$\texttt{Lambda-CDM-analytic-chi2.py}$', fontsize=20)
    plt.grid(True)
    # plt.show()

    # --- save fig ---
    fig.savefig('../thesis/figures/plots/EPS/Lambda-CDM-analytic-chi2_Omega-m0-vs-Omega-Lambda0-vs-chi2.eps', format='eps', bbox_inches='tight')
    fig.savefig('../thesis/figures/plots/PNG/Lambda-CDM-analytic-chi2_Omega-m0-vs-Omega-Lambda0-vs-chi2.png', format='png', bbox_inches='tight', dpi=250)
    fig.savefig('../thesis/figures/plots/PDF/Lambda-CDM-analytic-chi2_Omega-m0-vs-Omega-Lambda0-vs-chi2.pdf', format='pdf', bbox_inches='tight')
    # tikzplotlib.save('../thesis/figures/tikz/Lambda-CDM-analytic-chi2_Omega-m0-vs-Omega-Lambda0-vs-chi2.tex')


    # --- plot 2D contour ---
    fig, ax = plt.subplots()

    flat_line = plt.plot([0.0, 1.0], [1.0, 0.0], linestyle='--', color='grey')
    CP = ax.contour(X, Y, Z, levels=lvls, cmap='coolwarm_r', linewidths=1.0)
    plt.plot(Omega_m0_best, Omega_Lambda0_best, '.', color='red')   
    fmt = {}
    for l, s in zip(CP.levels, lvl_labels):
        fmt[l] = s
    ax.clabel(CP, inline=True, fmt=fmt)


    plt.xlabel(r'$\Omega_{\text{m},0}$', fontsize=12)
    plt.ylabel(r'$\Omega_{\Lambda,0}$', fontsize=12)
    # plt.suptitle(r'$\texttt{Lambda-CDM-analytic-chi2.py}$', fontsize=20)
    ax.tick_params(labelsize=10)
    plt.axis('scaled')
    plt.grid(True)

    text_location = np.array((0.51, 0.51))
    angle = 45
    trans_angle = plt.gca().transData.transform_angles(np.array((45,)), text_location.reshape((1, 2)))[0]
    ax.text(*text_location, r'flat universe ($\Omega_{k,0} = 0$)', rotation=-45, rotation_mode='anchor', transform_rotates_text=True, color='grey', fontsize=12)

    at = AnchoredText(fr'$\vb*{{\theta}}_{{\text{{best}}}} = ({Omega_m0_best:.2f}, {Omega_Lambda0_best:.2f})$', loc='upper right', borderpad=0.5, prop=dict(fontsize=12))
    at.patch.set(boxstyle='round,pad=0.2', fc='w', ec='0.5', alpha=0.9)
    ax.add_artist(at)

    # plt.show()

    # --- save fig ---
    fig.savefig('../thesis/figures/plots/EPS/Lambda-CDM-analytic-chi2_Omega-m0-vs-Omega-Lambda0.eps', format='eps', bbox_inches='tight')
    fig.savefig('../thesis/figures/plots/PNG/Lambda-CDM-analytic-chi2_Omega-m0-vs-Omega-Lambda0.png', format='png', bbox_inches='tight', dpi=250)
    fig.savefig('../thesis/figures/plots/PDF/Lambda-CDM-analytic-chi2_Omega-m0-vs-Omega-Lambda0.pdf', format='pdf', bbox_inches='tight')
    # tikzplotlib.save('../thesis/figures/tikz/Lambda-CDM-analytic-chi2_Omega-m0-vs-Omega-Lambda0.tex')
    # ================================================


if __name__ == "__main__":
    main()
