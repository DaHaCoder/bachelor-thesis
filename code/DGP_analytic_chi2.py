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
from scipy.integrate import quad, dblquad                          #   for integration -- https://docs.scipy.org/doc/scipy/tutorial/integrate.html
from scipy import optimize as opt                                  #   for optimization and fit -- https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
from scipy import stats                                            #   for chi2 -- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html#scipy.stats.chi2

### time package -- https://docs.python.org/3/library/time.html ###
import time                                                       #     for calculating computation time

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size':16})
# plt.rcParams['text.latex.preamble'] = r'''
# '''


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


def DGP_luminosity_distance(z, Omega_m0, alpha):
    # Cosmological Parameters
    # =======================
    c = 299792.458           # speed of light in vacuum in km/s
    # h = 0.6766               # Planck Collaboration 2018, Table 7, Planck+BAO -- https://www.aanda.org/articles/aa/full_html/2020/09/aa33880-18/T7.html
    # H_0 = h*100.0            # hubble constant in km/s per Mpc
    H_0 = 1.0                # dependence on hubble constant is set into the new_absolute_magnitude, see relative_magnitude(new_absolute_magnitude, new_luminosity_distance)
    d_H = c/H_0              # hubble distance
    # =======================
    
    sample_redshifts = np.linspace(0.0, max(z), 1000)
    sample_E = np.array([sol_friedmann(zi, Omega_m0, alpha, mod_friedmann, deriv_mod_friedmann) for zi in sample_redshifts])

    I = np.array([interp_integral(zi, sample_redshifts, sample_E) for zi in z])
    
    return (1.0 + z) * d_H * I


@njit
def relative_magnitude(new_absolute_magnitude, new_luminosity_distance):
    # new_luminosity_distance := H_0 * luminosity_distance
    # new_absolute_magnitude := old_absolute_magnitude - 5.0 * np.log10(H_0) + 25.0
    return new_absolute_magnitude + 5.0 * np.log10(new_luminosity_distance) 
# ====================== #


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


def chi2(Omega_m0, alpha, redshifts, magnitudes, error_magnitudes, zero_NaNs=False):
    new_absolute_magnitude = 0.0 
    d_L = DGP_luminosity_distance(redshifts, Omega_m0, alpha)
    rel_mag = relative_magnitude(new_absolute_magnitude, d_L)
    chi_2 = chi_square_analytic(magnitudes, error_magnitudes, rel_mag)
   
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



    # === Computation of chi2, finding best values for Omega_m0 and Omega_Lambda ===
    # ====================================================================================

    # --- define variables ---
    Omega_m0 = np.linspace(0.0, 0.4, 100)
    alpha    = np.linspace(0.0, 2.0, 100)
    
    # --- compute chi2 for every value in Omega_m0 and alpha ---
    MATRIX_ch2 = MATRIX_chi2(Omega_m0, alpha, redshifts, magnitudes, error_magnitudes)
    
    # --- find index of the values for Omega_m0 and alpha at which the chi2 has its maximum ---
    alpha_index, Omega_m0_index = find_best_fit_values(Omega_m0, alpha, MATRIX_ch2)
    Omega_m0_best = Omega_m0[Omega_m0_index]
    alpha_best = alpha[alpha_index]
    
    min_MATRIX_ch2 = MATRIX_ch2[alpha_index, Omega_m0_index]

    # --- compute sum_Omega_X0 ---
    sum_Omega_m0 = np.nansum(MATRIX_ch2, axis=0)
    sum_alpha = np.nansum(MATRIX_ch2, axis=1)

    # --- print best Omega_m0_best and alpha_best ---
    print("================================")
    print("Values for minimum chi2:")
    print(f"Omega_m0 = {Omega_m0_best:.3f}")
    print(f"alpha    = {alpha_best:.3f}")
    print("================================")
    # ====================================================================================
    
    END_TOTAL_TIME = time.perf_counter()
    print(f"********** TOTAL COMPUTATION TIME: {END_TOTAL_TIME - START_TOTAL_TIME:.2f} seconds **********")


    # === Plot Omega_m0 vs. chi2 for summed alpha and vice versa ===
    # ==============================================================

    fig = plt.figure()

    # --- plot Omega_m0 vs. chi2 for summed alpha ---
    plt.plot(Omega_m0, sum_Omega_m0, label='data')
    plt.plot(Omega_m0_best, min_MATRIX_ch2, 'o', color='red')
    plt.xlabel('$\Omega_{m,0}$')
    plt.ylabel('$\chi^2(\Omega_{m,0}, \sum \Omega_{\Lambda,0})$')
    plt.suptitle('$\\texttt{DGP_analytic_chi2.py}$', fontsize=20)
    plt.grid(True)
    plt.show()

    # --- save fig ---
    # fig.savefig('../thesis/figures/plots/EPS/[DGP_analytic_chi2]_Omega_m0_vs_chi2_at_alpha_best.eps', format = 'eps', bbox_inches = 'tight')
    fig.savefig('../thesis/figures/plots/PNG/[DGP_analytic_chi2]_Omega_m0_vs_chi2_at_alpha_best.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    # fig.savefig('../thesis/figures/plots/PDF/[DGP_analytic_chi2]_Omega_m0_vs_chi2_at_alpha_best.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/[DGP_analytic_chi2]_Omega_m0_vs_chi2_at_alpha_best.tex')

    # --- fig ---
    fig = plt.figure()
    
    # --- plot alpha vs. chi2 for summed Omega_m0 ---
    plt.plot(alpha, sum_alpha, label='data')
    plt.plot(alpha_best, min_MATRIX_ch2, 'o', color='red')
    plt.xlabel('$\Omega_{\Lambda,0}$')
    plt.ylabel('$\chi^2(\sum \Omega_{m,0}, \Omega_{\Lambda,0})$')
    plt.suptitle('$\\texttt{DGP_analytic_chi2.py}$', fontsize=20)
    plt.grid(True)
    plt.show()

    # --- save fig ---
    # fig.savefig('../thesis/figures/plots/EPS/[DGP_analytic_chi2]_alpha_vs_chi2_summed_Omega_m0.eps', format = 'eps', bbox_inches = 'tight')
    fig.savefig('../thesis/figures/plots/PNG/[DGP_analytic_chi2]_alpha_vs_chi2_summed_Omega_m0.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    # fig.savefig('../thesis/figures/plots/PDF/[DGP_analytic_chi2]_alpha_vs_chi2_summed_Omega_m0.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/[DGP_analytic_chi2]_alpha_vs_chi2_summed_Omega_m0.tex')
    # ==============================================================


   
    print(" === Omega_m0_best = ", Omega_m0_best)
    print(" === MATRIX_ch2[alpha_index, Omega_m0_index] = ", MATRIX_ch2[alpha_index, Omega_m0_index])

    # === Plot Omega_X0 vs. chi2 at Omega_Y0_best ===
    # ===============================================

    fig = plt.figure()

    # --- plot Omega_m0 vs. chi2 at alpha_best ---
    plt.plot(Omega_m0, MATRIX_ch2[alpha_index, :], label='data')
    plt.plot(Omega_m0_best, min_MATRIX_ch2, 'o', color='red')
    plt.xlabel('$\Omega_{m,0}$')
    plt.ylabel(f'$\\tilde{{\chi}}^2(\Omega_{{m,0}}, {alpha_best:.2f})$')
    plt.suptitle('$\\texttt{DGP_analytic_chi2.py}$', fontsize=20)
    plt.grid(True)
    plt.show()

    # --- save fig ---
    # fig.savefig('../thesis/figures/plots/EPS/[DGP_analytic_chi2]_Omega_m0_vs_chi2_at_alpha_best.eps', format = 'eps', bbox_inches = 'tight')
    fig.savefig('../thesis/figures/plots/PNG/[DGP_analytic_chi2]_Omega_m0_vs_chi2_at_alpha_best.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    # fig.savefig('../thesis/figures/plots/PDF/[DGP_analytic_chi2]_Omega_m0_vs_chi2_at_alpha_best.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/[DGP_analytic_chi2]_Omega_m0_vs_chi2_at_alpha_best.tex')

    # --- fig ---
    fig = plt.figure()
    
    # --- plot alpha vs. chi2 at Omega_m0_best ---
    plt.plot(alpha, MATRIX_ch2[:, Omega_m0_index], label='data')
    plt.plot(alpha_best, min_MATRIX_ch2, 'o', color='red')
    plt.xlabel('$\Omega_{m,0}$')
    plt.ylabel(f'$\\tilde{{\chi}}^2({Omega_m0_best:.2f}, \\alpha)$')
    plt.suptitle('$\\texttt{DGP_analytic_chi2.py}$', fontsize=20)
    plt.grid(True)
    plt.show()

    # --- save fig ---
    # fig.savefig('../thesis/figures/plots/EPS/[DGP_analytic_chi2]_alpha_vs_chi2_at_Omega_m0_best.eps', format = 'eps', bbox_inches = 'tight')
    fig.savefig('../thesis/figures/plots/PNG/[DGP_analytic_chi2]_alpha_vs_chi2_at_Omega_m0_best.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    # fig.savefig('../thesis/figures/plots/PDF/[DGP_analytic_chi2]_alpha_vs_chi2_at_Omega_m0_best.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/[DGP_analytic_chi2]_alpha_vs_chi2_at_Omega_m0_best.tex')
    # =======================================================================


    # === Plot Omega_m0 vs. alpha vs. chi2 ===
    # ======================================================

    # --- compute chi2 ---
    Z = np.array(MATRIX_ch2)
    X, Y = np.meshgrid(Omega_m0, alpha)

    conf_int = [stats.chi2.cdf(s**2.0, 1) for s in range(1,5)]
    lvls = [stats.chi2.ppf(ci, 2) + min_MATRIX_ch2 for ci in conf_int]
    lvl_labels = [f"${k}\sigma$".format(k) for k in range(1,5)]

    print("conf_int = ", conf_int)
    print("lvls = ", lvls)

    # --- plot 3D ---
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_wireframe(X, Y, Z, edgecolor='blue', alpha=0.3)
    ax.contour(X, Y, Z, zdir='z', levels=lvls, cmap='coolwarm')
    ax.set(xlabel='$\Omega_{m,0}$', ylabel='$\\alpha$', zlabel='$\\tilde{\chi}^2(\Omega_{m,0}, \\alpha)$')
    plt.suptitle('$\\texttt{DGP_analytic_chi2.py}$', fontsize=20)
    plt.grid(True)
    plt.show()

    # --- save fig ---
    # fig.savefig('../thesis/figures/plots/EPS/[DGP_analytic_chi2]_Omega_m0_vs_alpha_vs_chi2.eps', format = 'eps', bbox_inches = 'tight')
    fig.savefig('../thesis/figures/plots/PNG/[DGP_analytic_chi2]_Omega_m0_vs_alpha_vs_chi2.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    # fig.savefig('../thesis/figures/plots/PDF/[DGP_analytic_chi2]_Omega_m0_vs_alpha_vs_chi2.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/[DGP_analytic_chi2]_Omega_m0_vs_alpha_vs_chi2.tex')


    # --- plot 2D contour ---
    fig, ax = plt.subplots()

    CP = ax.contour(X, Y, Z, levels=lvls, cmap='coolwarm')
    
    fmt = {}
    for l, s in zip(CP.levels, lvl_labels):
        fmt[l] = s

    ax.clabel(CP, inline=True, fmt=fmt)

    plt.xlabel('$\Omega_{m,0}$')
    plt.ylabel('$\\alpha$')
    plt.suptitle('$\\texttt{DGP_analytic_chi2.py}$', fontsize=20)
    plt.grid(True)
    plt.show()

    # --- save fig ---
    # fig.savefig('../thesis/figures/plots/EPS/[DGP_analytic_chi2]_Omega_m0_vs_alpha.eps', format = 'eps', bbox_inches = 'tight')
    fig.savefig('../thesis/figures/plots/PNG/[DGP_analytic_chi2]_Omega_m0_vs_alpha.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    # fig.savefig('../thesis/figures/plots/PDF/[DGP_analytic_chi2]_Omega_m0_vs_alpha.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/[DGP_analytic_chi2]_Omega_m0_vs_alpha.tex')
    # ======================================================


if __name__ == "__main__":
    main()
