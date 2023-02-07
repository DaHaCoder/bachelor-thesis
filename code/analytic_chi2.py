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
from scipy import stats                                            #   for chi2 -- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html#scipy.stats.chi2

### time package -- https://docs.python.org/3/library/time.html ###
import time                                                       #     for calculating computation time

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size':16})
# plt.rcParams['text.latex.preamble'] = r'''
# '''

# def timeit(func):
#     def wrapper(*args, **kwargs):
#         start = time.perf_counter()
#         ret = func(*args, **kwargs)
#         end = time.perf_counter()
#         print(f"Time for {func.__name__}: {end - start} seconds")
#         return ret
#     return wrapper


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


@njit
def integrand(x, Omega_r0, Omega_m0, Omega_Lambda0):
    if x == 0.0:
        return 1.0
    Omega_K0 = 1.0 - Omega_r0 - Omega_m0 - Omega_Lambda0

    # define one_x := 1.0 + x and replace higher powers of it by recursive mulitiplications of one_x, since '**'-operator may be slow
    one_x = 1.0 + x
    one_x2 = one_x * one_x
    one_x3 = one_x2 * one_x
    one_x4 = one_x2 * one_x2
    return 1.0/np.sqrt(Omega_r0*one_x4 + Omega_m0*one_x3 + Omega_K0*one_x2 + Omega_Lambda0)


def integral(z, Omega_r0, Omega_m0, Omega_Lambda0):
    # d_C/d_H = Integrate[1/E(z'), {z',0,z}]
    return quad(integrand, 0.0, z, args=(Omega_r0, Omega_m0, Omega_Lambda0))[0]


def distances(z, Omega_m0, Omega_Lambda0):
    # Cosmological Parameters
    # =======================
    c = 299792.458
    # h = 0.6736
    # H_0 = h*100.0
    H_0 = 1.0
    Omega_r0 = 0.0
    d_H = c/H_0
    # =======================
    
    I = np.array([integral(zi, Omega_r0, Omega_m0, Omega_Lambda0) for zi in z])
    
    Omega_K0 = 1.0 - Omega_r0 - Omega_m0 - Omega_Lambda0
    # Omega_K0 = 0.0 

    if Omega_K0 > 0.0:
        transverse_comoving_distance = d_H * 1.0/np.sqrt(Omega_K0) * np.sinh(np.sqrt(Omega_K0) * I)

    elif Omega_K0 == 0.0:
        transverse_comoving_distance = d_H * I

    elif Omega_K0 < 0.0:
        transverse_comoving_distance = d_H * 1.0/np.sqrt(abs(Omega_K0)) * np.sin(np.sqrt(abs(Omega_K0)) * I)

    # angular_diameter_distance = 1.0/(1.0 + z)*transverse_comoving_distance
    luminosity_distance = (1.0 + z) * transverse_comoving_distance

    # return transverse_comoving_distance, angular_diameter_distance, luminosity_distance
    return luminosity_distance


@njit
def relative_magnitude(new_absolute_magnitude, new_luminosity_distance):
    # # luminosity_distance per Mpc, absolute_magnitude is at 10 pc (therefore + 25.0 since 10 pc = 10**(-5.0) Mpc)
    # return absolute_magnitude + 5.0*np.log10(luminosity_distance) + 25.0

    # h = 0.6736
    # H_0 = h*100.0
    # new_luminosity_distance := H_0 * luminosity_distance
    # new_absolute_magnitude := old_absolute_magnitude - 5.0 * np.log10(H_0) + 25.0
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


def chi2(Omega_m0, Omega_Lambda0, redshifts, magnitudes, error_magnitudes, zero_NaNs=False):
    new_absolute_magnitude = 0.0 
    d_L = distances(redshifts, Omega_m0, Omega_Lambda0)
    rel_mag = relative_magnitude(new_absolute_magnitude, d_L)
    chi_2 = chi_square_analytic(magnitudes, error_magnitudes, rel_mag)
   
    if np.isnan(chi_2) and zero_NaNs:
        chi_2 = 0.0

    return chi_2 


def chi2_par_helper(i, j, Omega_m0, Omega_Lambda0, redshifts, magnitudes, error_magnitudes):
    return i, j, chi2(Omega_m0[i], Omega_Lambda0[j], redshifts, magnitudes, error_magnitudes)


# @timeit("Compute normalization factor")
# def normalization_factor(redshifts, magnitudes, error_magnitudes, guess=1.0, bounds_m0=(0.0, 2.0), bounds_Lambda0=(-2.0, 2.0)):
#     L = lambda Omega_m0, Omega_Lambda0: chi2(Omega_m0, Omega_Lambda0, redshifts, magnitudes, error_magnitudes, L0=guess, zero_NaNs=True)
#     L0 = guess / dblquad(L, *bounds_Lambda0, *bounds_m0)[0]
#     return L0


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
    # FIXME: There are some NaN's in the matrix...
    return np.unravel_index(np.nanargmin(MATRIX_chi2), MATRIX_chi2.shape)
    
    

# def chi2_fit(x, a, b, c):
#     return a * stats.chi2.ppf(x - b, 1) + c 


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



    # === Computation of chi2, finding best values for Omega_m0 and Omega_Lambda ===
    # ====================================================================================

    # --- define variables ---
    Omega_m0      = np.linspace(0.0, 1.0, 100)
    Omega_Lambda0 = np.linspace(0.0, 1.2, 100)
    
    # --- compute chi2 for every value in Omega_m0 and Omega_Lambda0 ---
    MATRIX_ch2 = MATRIX_chi2(Omega_m0, Omega_Lambda0, redshifts, magnitudes, error_magnitudes)
    
    # --- find index of the values for Omega_m0 and Omega_Lambda0 at which the chi2 has its maximum ---
    Omega_Lambda0_index, Omega_m0_index = find_best_fit_values(Omega_m0, Omega_Lambda0, MATRIX_ch2)
    Omega_m0_best = Omega_m0[Omega_m0_index]
    Omega_Lambda0_best = Omega_Lambda0[Omega_Lambda0_index]
    
    min_MATRIX_ch2 = MATRIX_ch2[Omega_Lambda0_index, Omega_m0_index]

    # --- compute sum_Omega_X0 ---
    sum_Omega_m0 = np.nansum(MATRIX_ch2, axis=0)
    sum_Omega_Lambda0 = np.nansum(MATRIX_ch2, axis=1)

    # --- print best Omega_m0_best and Omega_Lambda0_best ---
    print("================================")
    print("Values for minimum chi2:")
    print(f"Omega_m0      = {Omega_m0_best:.3f}")
    print(f"Omega_Lambda0 = {Omega_Lambda0_best:.3f}")
    print("================================")
    # ====================================================================================
    
    END_TOTAL_TIME = time.perf_counter()
    print(f"********** TOTAL COMPUTATION TIME: {END_TOTAL_TIME - START_TOTAL_TIME:.2f} seconds **********")
    
    # # === Plot Omega_X0 vs. chi2 for summed Omega_Y0 ===
    # # ==================================================

    # fig = plt.figure()

    # # --- plot Omega_m0 vs. chi2 for summed Omega_Lambda0 ---
    # plt.plot(Omega_m0, sum_Omega_m0, label='data')
    # plt.xlabel('$\Omega_{m,0}$')
    # plt.ylabel('$\chi^2(\Omega_{m,0}, \sum \Omega_{\Lambda,0})$')
    # plt.suptitle('$\\texttt{analytic_chi2.py}$', fontsize=20)
    # plt.grid(True)
    # plt.show()

    # # --- save fig ---
    # # fig.savefig('../thesis/figures/plots/EPS/[analytic_chi2]_Omega_m0_vs_chi2_at_Omega_Lambda0_best.eps', format = 'eps', bbox_inches = 'tight')
    # fig.savefig('../thesis/figures/plots/PNG/[analytic_chi2]_Omega_m0_vs_chi2_at_Omega_Lambda0_best.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    # # fig.savefig('../thesis/figures/plots/PDF/[analytic_chi2]_Omega_m0_vs_chi2_at_Omega_Lambda0_best.pdf', format = 'pdf', bbox_inches = 'tight')
    # # tikzplotlib.save('../thesis/figures/tikz/[analytic_chi2]_Omega_m0_vs_chi2_at_Omega_Lambda0_best.tex')

    # # --- fig ---
    # fig = plt.figure()
    
    # # --- plot Omega_Lambda0 vs. chi2 for summed Omega_m0 ---
    # plt.plot(Omega_Lambda0, sum_Omega_Lambda0, label='data')
    # plt.xlabel('$\Omega_{\Lambda,0}$')
    # plt.ylabel('$\chi^2(\sum \Omega_{m,0}, \Omega_{\Lambda,0})$')
    # plt.suptitle('$\\texttt{analytic_chi2.py}$', fontsize=20)
    # plt.grid(True)
    # plt.show()

    # # --- save fig ---
    # # fig.savefig('../thesis/figures/plots/EPS/[analytic_chi2]_Omega_Lambda0_vs_chi2_summed_Omega_m0.eps', format = 'eps', bbox_inches = 'tight')
    # fig.savefig('../thesis/figures/plots/PNG/[analytic_chi2]_Omega_Lambda0_vs_chi2_summed_Omega_m0.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    # # fig.savefig('../thesis/figures/plots/PDF/[analytic_chi2]_Omega_Lambda0_vs_chi2_summed_Omega_m0.pdf', format = 'pdf', bbox_inches = 'tight')
    # # tikzplotlib.save('../thesis/figures/tikz/[analytic_chi2]_Omega_Lambda0_vs_chi2_summed_Omega_m0.tex')
    # # =======================================================================


   
    print(" === Omega_m0_best = ", Omega_m0_best)
    print(" === MATRIX_ch2[Omega_Lambda0_index, Omega_m0_index] = ", MATRIX_ch2[Omega_Lambda0_index, Omega_m0_index])

    # === Plot Omega_X0 vs. chi2 at Omega_Y0_best ===
    # ===============================================

    fig = plt.figure()

    # --- plot Omega_m0 vs. chi2 at Omega_Lambda0_best ---
    plt.plot(Omega_m0, MATRIX_ch2[Omega_Lambda0_index, :], label='data')
    plt.plot(Omega_m0_best, min_MATRIX_ch2, 'o', color='red')
    plt.xlabel('$\Omega_{m,0}$')
    plt.ylabel(f'$\chi^2(\Omega_{{m,0}}, {Omega_Lambda0_best:.2f})$')
    plt.suptitle('$\\texttt{analytic_chi2.py}$', fontsize=20)
    plt.grid(True)
    plt.show()

    # --- save fig ---
    # fig.savefig('../thesis/figures/plots/EPS/[analytic_chi2]_Omega_m0_vs_chi2_at_Omega_Lambda0_best.eps', format = 'eps', bbox_inches = 'tight')
    fig.savefig('../thesis/figures/plots/PNG/[analytic_chi2]_Omega_m0_vs_chi2_at_Omega_Lambda0_best.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    # fig.savefig('../thesis/figures/plots/PDF/[analytic_chi2]_Omega_m0_vs_chi2_at_Omega_Lambda0_best.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/[analytic_chi2]_Omega_m0_vs_chi2_at_Omega_Lambda0_best.tex')

    # --- fig ---
    fig = plt.figure()
    
    # --- plot Omega_Lambda0 vs. chi2 at Omega_m0_best ---
    plt.plot(Omega_Lambda0, MATRIX_ch2[:, Omega_m0_index], label='data')
    plt.plot(Omega_Lambda0_best, min_MATRIX_ch2, 'o', color='red')
    plt.xlabel('$\Omega_{m,0}$')
    plt.ylabel(f'$\chi^2({Omega_m0_best:.2f}, \Omega_{{\Lambda,0}}$')
    plt.suptitle('$\\texttt{analytic_chi2.py}$', fontsize=20)
    plt.grid(True)
    plt.show()

    # --- save fig ---
    # fig.savefig('../thesis/figures/plots/EPS/[analytic_chi2]_Omega_Lambda0_vs_chi2_at_Omega_m0_best.eps', format = 'eps', bbox_inches = 'tight')
    fig.savefig('../thesis/figures/plots/PNG/[analytic_chi2]_Omega_Lambda0_vs_chi2_at_Omega_m0_best.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    # fig.savefig('../thesis/figures/plots/PDF/[analytic_chi2]_Omega_Lambda0_vs_chi2_at_Omega_m0_best.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/[analytic_chi2]_Omega_Lambda0_vs_chi2_at_Omega_m0_best.tex')
    # =======================================================================


    # === Plot Omega_m0 vs. Omega_Lambda0 vs. chi2 ===
    # ======================================================

    # --- compute chi2 ---
    Z = np.array(MATRIX_ch2)
    X, Y = np.meshgrid(Omega_m0, Omega_Lambda0)

    conf_int = [stats.chi2.cdf(s**2.0, 1) for s in range(1,5)]
    lvls = [stats.chi2.ppf(ci, 2) + min_MATRIX_ch2 for ci in conf_int]

    print("conf_int = ", conf_int)
    print("lvls = ", lvls)

    # --- plot 3D ---
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_wireframe(X, Y, Z, edgecolor='blue', alpha=0.3)
    ax.contourf(X, Y, Z, zdir='z', levels=lvls, cmap='coolwarm')
    plt.xlabel('$\Omega_{m,0}$')
    plt.ylabel('$\Omega_{\Lambda,0}$')
    plt.suptitle('$\\texttt{analytic_chi2.py}$', fontsize=20)
    plt.grid(True)
    plt.show()

    # --- save fig ---
    # fig.savefig('../thesis/figures/plots/EPS/[analytic_chi2]_Omega_m0_vs_Omega_Lambda0_vs_chi2.eps', format = 'eps', bbox_inches = 'tight')
    fig.savefig('../thesis/figures/plots/PNG/[analytic_chi2]_Omega_m0_vs_Omega_Lambda0_vs_chi2.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    # fig.savefig('../thesis/figures/plots/PDF/[analytic_chi2]_Omega_m0_vs_Omega_Lambda0_vs_chi2.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/[analytic_chi2]_Omega_m0_vs_Omega_Lambda0_vs_chi2.tex')


    # --- plot 2D contour ---
    fig, ax = plt.subplots()

    ax.contour(X, Y, Z, levels=lvls, cmap='coolwarm')
    plt.xlabel('$\Omega_{m,0}$')
    plt.ylabel('$\Omega_{\Lambda,0}$')
    plt.suptitle('$\\texttt{analytic_chi2.py}$', fontsize=20)
    plt.grid(True)
    plt.show()

    # --- save fig ---
    # fig.savefig('../thesis/figures/plots/EPS/[analytic_chi2]_Omega_m0_vs_Omega_Lambda0.eps', format = 'eps', bbox_inches = 'tight')
    fig.savefig('../thesis/figures/plots/PNG/[analytic_chi2]_Omega_m0_vs_Omega_Lambda0.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    # fig.savefig('../thesis/figures/plots/PDF/[analytic_chi2]_Omega_m0_vs_Omega_Lambda0.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/[analytic_chi2]_Omega_m0_vs_Omega_Lambda0.tex')
    # ======================================================


if __name__ == "__main__":
    main()
