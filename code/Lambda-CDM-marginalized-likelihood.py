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
def chi_square(magnitudes, error_magnitudes, theoretical_magnitudes):
    return np.sum(np.square((theoretical_magnitudes - magnitudes) / error_magnitudes))


@njit
def l(mod_absolute_magnitude, mod_luminosity_distance, magnitudes, error_magnitudes, L0):
    m_th = theoretical_magnitude(mod_absolute_magnitude, mod_luminosity_distance)
    chi_2 = chi_square(magnitudes, error_magnitudes, m_th)
    return L0 * np.exp(-0.5 * chi_2)


def marginalized_likelihood(Omega_m0, Omega_Lambda0, redshifts, magnitudes, error_magnitudes, L0, zero_NaNs=False):
    D_L = mod_luminosity_distance(redshifts, Omega_m0, Omega_Lambda0)
    min_absolute_magnitude = 15.0
    max_absolute_magnitude = 17.0
    margin_L = quad(l, min_absolute_magnitude, max_absolute_magnitude, args=(D_L, magnitudes, error_magnitudes, L0))[0]
    if np.isnan(margin_L) and zero_NaNs:
        margin_L = 0.0
    return margin_L


def marginalized_likelihood_par_helper(i, j, Omega_m0, Omega_Lambda0, redshifts, magnitudes, error_magnitudes, L0):
    return i, j, marginalized_likelihood(Omega_m0[i], Omega_Lambda0[j], redshifts, magnitudes, error_magnitudes, L0)


@timeit("Compute normalization factor")
def normalization_factor(redshifts, magnitudes, error_magnitudes, guess=1.0, bounds_Omega_m0=(0.0, 2.5), bounds_Omega_Lambda0=(-2.5, 2.5)):
    L = lambda Omega_m0, Omega_Lambda0: marginalized_likelihood(Omega_m0, Omega_Lambda0, redshifts, magnitudes, error_magnitudes, L0=guess, zero_NaNs=True)
    L0 = guess / dblquad(L, *bounds_Omega_Lambda0, *bounds_Omega_m0)[0]
    return L0


@timeit("Compute likelihood on grid")
def MATRIX_marginalized_likelihood(Omega_m0, Omega_Lambda0, redshifts, magnitudes, error_magnitudes, L0):
    rows = len(Omega_m0)
    cols = len(Omega_Lambda0)
    
    # define matrix where 'rows' is the amount of rows and 'cols' the amount of columns
    MATRIX = np.zeros((rows, cols))
    marginalized_likelihood_args = ((i, j, Omega_m0, Omega_Lambda0, redshifts, magnitudes, error_magnitudes, L0) for i, j in product(range(rows), range(cols)))

    with Pool() as pool:
        for i, j, L in pool.starmap(marginalized_likelihood_par_helper, marginalized_likelihood_args):
            MATRIX[i, j] = L

    # for args in likelihood_args:
    #     i, j, L = new_likelihood(*args)
    #     MATRIX[i, j] = L

    return MATRIX.T


def find_best_fit_values(Omega_m0, Omega_Lambda0, MATRIX_marg_likelihood):
    return np.unravel_index(np.nanargmax(MATRIX_marg_likelihood), MATRIX_marg_likelihood.shape)


def gauss_curve(x, mu, sigma, y0):
    return y0 * np.exp(-(x - mu) * (x - mu)/(2.0 * sigma * sigma))


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



    # === Computation of marginalized likelihood, finding best values for Omega_m0 and Omega_Lambda ===
    # =================================================================================================
    
    # --- define variables ---
    Omega_m0      = np.linspace(0.0, 1.0, 200)
    Omega_Lambda0 = np.linspace(0.0, 1.2, 200)

    # --- compute normalization factor for marginalized likelihood ---
    L0 = normalization_factor(redshifts, magnitudes, error_magnitudes, guess=1e+124)
    # L0 = 3.022013e+125

    # --- compute marginalized likelihood for every value in Omega_m0 and Omega_Lambda0 ---
    MATRIX_marg_like = MATRIX_marginalized_likelihood(Omega_m0, Omega_Lambda0, redshifts, magnitudes, error_magnitudes, L0)

    print()
    print("==========================")
    print(f"{L0 = :.6e}")
    print(f"Sanity check: Riemann integral of likelihoods = {np.nansum(MATRIX_marg_like) * np.diff(Omega_m0)[0] * np.diff(Omega_Lambda0)[0]:.6f}")
    print()

    # --- find index of the values for Omega_m0 and Omega_Lambda0 where MATRIX_marg_like has its maximum ---
    Omega_Lambda0_index, Omega_m0_index = find_best_fit_values(Omega_m0, Omega_Lambda0, MATRIX_marg_like)
    Omega_m0_best = Omega_m0[Omega_m0_index]
    Omega_Lambda0_best = Omega_Lambda0[Omega_Lambda0_index]

    # --- likelihood at its maximum ---
    max_MATRIX_marg_like = MATRIX_marg_like[Omega_Lambda0_index, Omega_m0_index]

    # --- compute MATRIX_marg_like_summed_Omega_X0 depending on Omega_Y ---
    MATRIX_marg_like_summed_Omega_Lambda0 = np.nansum(MATRIX_marg_like, axis=0)
    MATRIX_marg_like_summed_Omega_m0 = np.nansum(MATRIX_marg_like, axis=1)
    # COMMENT MATRIX_marg_like_summed_Omega_X0 = np.nansum(MATRIX_marg_like, axis=0)
    # for i in range(len(Omega_X0)):
    #    s = 0.0
    #    for j in range(len(Omega_Y0)):
    #        s += MATRIX_marg_like[i, j]
    #    MATRIX_marg_like_summed_Omega_X0[i] = s
    
    # --- parameter guess for gauss fits p0_guess_X0 = [mu, sigma, y0] ---
    p0_guess_m0 = [Omega_m0_best, 0.1, 1.0]
    p0_guess_Lambda0 = [Omega_Lambda0_best, 0.1, 1.0]
    
    # --- calculate parameters by using scipy optimize with defined gauss_curve ---
    # (_, sigma_m0, _), *_      = opt.curve_fit(gauss_curve, Omega_m0, MATRIX_marg_like_summed_Omega_m0, p0_guess_m0)
    # (_, sigma_Lambda0, _), *_ = opt.curve_fit(gauss_curve, Omega_Lambda0, MATRIX_marg_like_summed_Omega_Lambda0, p0_guess_Lambda0)
    
    # --- calculate parameters by using scipy optimize with defined gauss_curve ---
    popt_m0, pcov_m0 = opt.curve_fit(gauss_curve, Omega_m0, MATRIX_marg_like_summed_Omega_Lambda0, p0_guess_m0)
    popt_Lambda0, pcov_Lambda0 = opt.curve_fit(gauss_curve, Omega_Lambda0, MATRIX_marg_like_summed_Omega_m0, p0_guess_Lambda0)
    
    mu_m0, sigma_m0, y0_m0 = popt_m0
    mu_Lambda0, sigma_Lambda0, y0_Lambda0 = popt_Lambda0

    sigma_m0 = abs(sigma_m0)
    sigma_Lambda0 = abs(sigma_Lambda0)
    
    # --- print parameters for gauss fit ---
    # print("=== Parameters for gauss fit ===")
    # print("================================")
    # print("mu_m0, sigma_m0, y0_m0 = ",  popt_m0)
    # print("mu_Lambda0, sigma_Lambda0, y0_Lambda0 = ", popt_Lambda0)
    # print("================================")
   
    # --- compute fitted gauss curves ---
    MATRIX_marg_like_summed_Omega_Lambda0_gauss_fit = gauss_curve(Omega_m0, *popt_m0)
    MATRIX_marg_like_summed_Omega_m0_gauss_fit = gauss_curve(Omega_Lambda0, *popt_Lambda0)

    # --- print best Omega_m0_best and Omega_Lambda0_best ---
    print("==================================")
    print("Cosmological Parameter Estimation:")
    print("----------------------------------")
    print(f"Omega_m0      = {Omega_m0_best:.3f} ± {sigma_m0:.3f}")
    print(f"Omega_Lambda0 = {Omega_Lambda0_best:.3f} ± {sigma_Lambda0:.3f}")
    print("==================================")
    # =================================================================================================


    END_TOTAL_TIME = time.perf_counter()
    print(f"********** TOTAL COMPUTATION TIME: {END_TOTAL_TIME - START_TOTAL_TIME:.2f} seconds **********")


    # === Plot Omega_X0 vs. marginalized likelihood for summed Omega_Y0 with gauss fit ===
    # ====================================================================================

    # --- plot Omega_m0 vs. marginalized likelihood for summed Omega_Lambda0 with gauss fit ---
    fig, ax = plt.subplots()

    # at = AnchoredText(fr'$\vb*{{\theta_{{\text{{best}}}}}} = ({Omega_m0_best:.2f}, {Omega_Lambda0_best:.2f})$', loc='upper right', borderpad=0.5)
    # at = AnchoredText(fr'$(\Omega_{{\text{{m}}, 0, \text{{best}}}}, \Omega_{{\Lambda, 0, \text{{best}}}}) \pm (\sigma_{{\text{{m}}, 0}}, \sigma_{{\Lambda, 0}}) = ({Omega_m0_best:.2f}, {Omega_Lambda0_best:.2f}) \pm ({sigma_m0:.2f}, {sigma_Lambda0:.2f})$', loc='upper right', borderpad=0.5)
    # at.patch.set(boxstyle='round,pad=0.2', fc='w', ec='0.5', alpha=0.9)
    # ax.add_artist(at)
   
    plt.plot(Omega_m0, MATRIX_marg_like_summed_Omega_Lambda0, label='data')
    plt.plot(Omega_m0, MATRIX_marg_like_summed_Omega_Lambda0_gauss_fit, linestyle='--', color='tab:orange', label='fit')
    plt.xlabel(r'$\Omega_{\text{m},0}$', fontsize=16)
    plt.ylabel(r'$L_{\text{M}, \sum \Omega_{\Lambda,0}}(\Omega_{\text{m},0} \vert D)$', fontsize=16)
    # plt.suptitle(r'$\texttt{Lambda-CDM-marginalized-likelihood.py}$', fontsize=20)
    plt.title(fr'$(\Omega_{{\text{{m}}, 0, \text{{best}}}}, \Omega_{{\Lambda, 0, \text{{best}}}}) \pm (\sigma_{{\text{{m}}, 0}}, \sigma_{{\Lambda, 0}}) = ({Omega_m0_best:.2f}, {Omega_Lambda0_best:.2f}) \pm ({sigma_m0:.2f}, {sigma_Lambda0:.2f})$')
    plt.grid(True) 
    # plt.show()
    
    # --- save fig ---
    fig.savefig('../thesis/figures/plots/EPS/Lambda-CDM-marginalized-likelihood_Omega-m0-vs-likelihood-summed-Omega-Lambda0.eps', format='eps', bbox_inches='tight')
    fig.savefig('../thesis/figures/plots/PNG/Lambda-CDM-marginalized-likelihood_Omega-m0-vs-likelihood-summed-Omega-Lambda0.png', format='png', bbox_inches='tight', dpi=250)
    fig.savefig('../thesis/figures/plots/PDF/Lambda-CDM-marginalized-likelihood_Omega-m0-vs-likelihood-summed-Omega-Lambda0.pdf', format='pdf', bbox_inches='tight')
    # tikzplotlib.save('../thesis/figures/tikz/Lambda-CDM-marginalized-likelihood_Omega-m0-vs-likelihood-summed-Omega-Lambda0.tex')
   
    # --- plot Omega_Lambda0 vs. marginalized likelihood for summed Omega_m0 with gauss fit ---
    fig, ax = plt.subplots()

    # at = AnchoredText(fr'$\vb*{{\theta_{{\text{{best}}}}}} = ({Omega_m0_best:.2f}, {Omega_Lambda0_best:.2f})$', loc='upper left', borderpad=0.5)
    # at = AnchoredText(fr'$(\Omega_{{\text{{m}}, 0, \text{{best}}}}, \Omega_{{\Lambda, 0, \text{{best}}}}) \pm (\sigma_{{\text{{m}}, 0}}, \sigma_{{\Lambda, 0}}) = ({Omega_m0_best:.2f}, {Omega_Lambda0_best:.2f}) \pm ({sigma_m0:.2f}, {sigma_Lambda0:.2f})$', loc='upper left', borderpad=0.5)
    # at.patch.set(boxstyle='round,pad=0.2', fc='w', ec='0.5', alpha=0.9)
    # ax.add_artist(at)
    
    plt.plot(Omega_Lambda0, MATRIX_marg_like_summed_Omega_m0, label='data')
    plt.plot(Omega_Lambda0, MATRIX_marg_like_summed_Omega_m0_gauss_fit, linestyle='--', color='tab:orange', label='fit')
    plt.xlabel(r'$\Omega_{\Lambda,0}$')
    plt.ylabel(r'$L_{\text{M}, \sum \Omega_{\text{m},0}}(\Omega_{\Lambda,0} \vert D)$')
    # plt.suptitle(r'$\texttt{Lambda-CDM-marginalized-likelihood.py}$', fontsize=20)
    plt.title(fr'$(\Omega_{{\text{{m}}, 0, \text{{best}}}}, \Omega_{{\Lambda, 0, \text{{best}}}}) \pm (\sigma_{{\text{{m}}, 0}}, \sigma_{{\Lambda, 0}}) = ({Omega_m0_best:.2f}, {Omega_Lambda0_best:.2f}) \pm ({sigma_m0:.2f}, {sigma_Lambda0:.2f})$')
    plt.grid(True) 
    # plt.show()

    # --- save fig ---
    fig.savefig('../thesis/figures/plots/EPS/Lambda-CDM-marginalized-likelihood_Omega-Lambda0-vs-likelihood-summed-Omega-m0.eps', format='eps', bbox_inches='tight')
    fig.savefig('../thesis/figures/plots/PNG/Lambda-CDM-marginalized-likelihood_Omega-Lambda0-vs-likelihood-summed-Omega-m0.png', format='png', bbox_inches='tight', dpi=250)
    fig.savefig('../thesis/figures/plots/PDF/Lambda-CDM-marginalized-likelihood_Omega-Lambda0-vs-likelihood-summed-Omega-m0.pdf', format='pdf', bbox_inches='tight')
    # tikzplotlib.save('../thesis/figures/tikz/Lambda-CDM-marginalized-likelihood_Omega-Lambda0-vs-likelihood-summed-Omega-m0.tex')
    # ================================================================================================


    # === Plot Omega_m0 vs. Omega_Lambda0 vs. marginalized likelihood ===
    # ===================================================================
    
    # --- compute likelihood ---
    Z = np.array(MATRIX_marg_like)
    X, Y = np.meshgrid(Omega_m0, Omega_Lambda0)

    conf_int = [stats.chi2.cdf(s**2.0, 1) for s in range(1,5)]
    lvls = [max_MATRIX_marg_like * np.exp(-0.5 * stats.chi2.ppf(ci, 2)) for ci in conf_int]
    lvls.sort()
    lvl_labels = [f'${k}\sigma$'.format(k) for k in reversed(range(1,5))]

    # --- plot 3D ---
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_wireframe(X, Y, Z, edgecolor='tab:blue', alpha=0.3)
    ax.contour(X, Y, Z, zdir='z', levels=lvls, offset=0, cmap='coolwarm')
    ax.scatter(Omega_m0_best, Omega_Lambda0_best, 0.0, 'o', color='red')
    ax.set(xlabel=r'$\Omega_{\text{m},0}$', ylabel=r'$\Omega_{\Lambda,0}$', zlabel=r'$L_{\text{M}}(\Omega_{\text{m},0}, \Omega_{\Lambda,0} \vert D)$')
    # plt.suptitle(r'$\texttt{Lambda-CDM-marginalized-likelihood.py}$', fontsize=20)
    # plt.title(rf'best fit values: $(\Omega_{{\text{{m}},0}}, \Omega_{{\Lambda,0}}) \pm (\sigma_{{\text{{m}},0}}, \sigma_{{\Lambda,0}}) = ({Omega_m0_best:.2f}, {Omega_Lambda0_best:.2f}) \pm ({sigma_m0:.2f}, {sigma_Lambda0:.2f})$')
    plt.grid(True)
    # plt.show()

    # --- save fig ---
    fig.savefig('../thesis/figures/plots/EPS/Lambda-CDM-marginalized-likelihood_Omega-m0-vs-Omega-Lambda0-vs-likelihood.eps', format='eps', bbox_inches='tight')
    fig.savefig('../thesis/figures/plots/PNG/Lambda-CDM-marginalized-likelihood_Omega-m0-vs-Omega-Lambda0-vs-likelihood.png', format='png', bbox_inches='tight', dpi=250)
    fig.savefig('../thesis/figures/plots/PDF/Lambda-CDM-marginalized-likelihood_Omega-m0-vs-Omega-Lambda0-vs-likelihood.pdf', format='pdf', bbox_inches='tight')
    # tikzplotlib.save('../thesis/figures/tikz/Lambda-CDM-marginalized-likelihood_Omega-m0-vs-Omega-Lambda0-vs-likelihood.tex')
    
    # --- plot 2D contour ---
    fig, ax = plt.subplots()
    
    flat_line = plt.plot([0.0, 1.0], [1.0, 0.0], linestyle='--', color='grey')
    CP = ax.contour(X, Y, Z, levels=lvls, cmap='coolwarm')
    plt.plot(Omega_m0_best, Omega_Lambda0_best, 'o', color='red')
    fmt = {}
    for l, s in zip(CP.levels, lvl_labels):
        fmt[l] = s
    ax.clabel(CP, inline=True, fmt=fmt)

    text_location = np.array((0.51, 0.51))
    angle = 45
    trans_angle = plt.gca().transData.transform_angles(np.array((45,)), text_location.reshape((1, 2)))[0]
    ax.text(*text_location, r'flat universe ($\Omega_{k,0} = 0$)', rotation=-45, rotation_mode='anchor', transform_rotates_text=True, color='grey', fontsize=20)

    # at = AnchoredText(fr'$\vb*{{\theta_{{\text{{best}}}}}} = ({Omega_m0_best:.2f}, {Omega_Lambda0_best:.2f})$', loc='lower left', borderpad=0.5)
    # at = AnchoredText(fr'$(\Omega_{{\text{{m}}, 0, \text{{best}}}}, \Omega_{{\Lambda, 0, \text{{best}}}}) \pm (\sigma_{{\text{{m}}, 0}}, \sigma_{{\Lambda, 0}}) = ({Omega_m0_best:.2f}, {Omega_Lambda0_best:.2f}) \pm ({sigma_m0:.2f}, {sigma_Lambda0:.2f})$', loc='lower left', borderpad=0.5)
    # at.patch.set(boxstyle='round,pad=0.2', fc='w', ec='0.5', alpha=0.9)
    # ax.add_artist(at)

    plt.xlabel(r'$\Omega_{\text{m},0}$', fontsize=16) 
    plt.ylabel(r'$\Omega_{\Lambda,0}$', fontsize=16)
    # plt.suptitle(r'$\texttt{Lambda-CDM-marginalized-likelihood.py}$', fontsize=20)
    # plt.title(rf'best fit values: $(\Omega_{{\text{{m}},0}}, \Omega_{{\Lambda,0}}) \pm (\sigma_{{\text{{m}},0}}, \sigma_{{\Lambda,0}}) = ({Omega_m0_best:.2f}, {Omega_Lambda0_best:.2f}) \pm ({sigma_m0:.2f}, {sigma_Lambda0:.2f})$')
    plt.grid(True)
    # plt.show()

    # --- save fig ---
    fig.savefig('../thesis/figures/plots/EPS/Lambda-CDM-marginalized-likelihood_Omega-m0-vs-Omega-Lambda0.eps', format='eps', bbox_inches='tight')
    fig.savefig('../thesis/figures/plots/PNG/Lambda-CDM-marginalized-likelihood_Omega-m0-vs-Omega-Lambda0.png', format='png', bbox_inches='tight', dpi=250)
    fig.savefig('../thesis/figures/plots/PDF/Lambda-CDM-marginalized-likelihood_Omega-m0-vs-Omega-Lambda0.pdf', format='pdf', bbox_inches='tight')
    # tikzplotlib.save('../thesis/figures/tikz/Lambda-CDM-marginalized-likelihood_Omega-m0-vs-Omega-Lambda0.tex')
    # ===================================================================


if __name__ == "__main__":
    main()
