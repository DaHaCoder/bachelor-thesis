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
    # D_C/D_H = Integrate[1/E(z'), {z',0,z}]
    return quad(integrand, 0.0, z, args=(Omega_r0, Omega_m0, Omega_Lambda0))[0]


def distances(z, Omega_m0, Omega_Lambda0):
    # Cosmological Parameters
    # =======================
    c = 299792.458
    # h = 0.6736
    # H_0 = h*100
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

    # angular_diameter_distance = 1.0/(1.0 + z) * transverse_comoving_distance
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
    # new_absolute_magnitude := absolute_magnitude - 5.0 * np.log10(H_0) + 25.0
    return new_absolute_magnitude + 5.0 * np.log10(new_luminosity_distance) 


@njit
def chi_square(magnitudes, error_magnitudes, relative_magnitudes):
    return np.sum(np.square((relative_magnitudes - magnitudes) / error_magnitudes))


@njit
def likelihood(new_absolute_magnitude, luminosity_distance, magnitudes, error_magnitudes):
    mag = relative_magnitude(new_absolute_magnitude, luminosity_distance)
    chi_2 = chi_square(magnitudes, error_magnitudes, mag)
    return np.exp(-0.5 * chi_2)


def marginalized_likelihood(i, j, Omega_m0, Omega_Lambda0, redshifts, magnitudes, error_magnitudes):
    d_L = distances(redshifts, Omega_m0[i], Omega_Lambda0[j])
    min_absolute_magnitude = 15.0 
    max_absolute_magnitude = 17.0 
    margin_L = quad(likelihood, min_absolute_magnitude, max_absolute_magnitude, args=(d_L, magnitudes, error_magnitudes))[0]
    return i, j, margin_L


# def normalization_factor(Omega_r0, Omega_m0, Omega_Lambda0, c, H_0, redshifts, magnitudes, error_magnitudes):
    # L = lambda Omega_m0, Omega_Lambda0, absolute_magnitude: likelihood(Omega_r0, Omega_m0, Omega_Lambda0, c, H_0, absolute_magnitude, redshifts, magnitudes, error_magnitudes)
    # x_1 = lambda Omega_Lambda0, absolute_magnitude: min(LIST_Omega_m0)
    # x_2 = lambda Omega_Lambda0, absolute_magnitude: max(LIST_Omega_m0)
    # y_1 = lambda absolute_magnitude: min(LIST_Omega_Lambda0)
    # y_2 = lambda absolute_magnitude: max(LIST_Omega_Lambda0)
    # z_1 = -1.0
    # z_2 = 1.0
    # L0 = tplquad(L, z_1, z_2, y_1, y_2, x_1, x_2)[0]
    # L = lambda Omega_m0, Omega_Lambda0: marginalized_likelihood(i, j, Omega_r0, Omega_m0, Omega_Lambda0, c, H_0, redshifts, magnitudes, error_magnitudes)
    # min_Omega_m0 = lambda Omega_Lambda0: min(LIST_Omega_m0)
    # max_Omega_m0 = lambda Omega_Lambda0: max(LIST_Omega_m0)
    # min_Omega_Lambda0 = min(LIST_Omega_Lambda0)
    # max_Omega_Lambda0 = max(LIST_Omega_Lambda0)
    # L0 = dblquad(L, min_Omega_Lambda0, max_Omega_Lambda0, min_Omega_m0, max_Omega_m0)[0]
    # print('L0 = ', L0)
    # return L0


def MATRIX_likelihood(LIST_Omega_m0, LIST_Omega_Lambda0, redshifts, magnitudes, error_magnitudes):
    rows = len(LIST_Omega_m0)
    cols = len(LIST_Omega_Lambda0)
    
    # define matrix where 'rows' is the amount of rows and 'cols' the amount of columns
    MATRIX = np.zeros((rows, cols))

    likelihood_args = ((i, j, LIST_Omega_m0, LIST_Omega_Lambda0, redshifts, magnitudes, error_magnitudes) for i, j in product(range(rows), range(cols)))

    # L0 = normalization_factor(Omega_r0, LIST_Omega_m0, LIST_Omega_Lambda0, c, H_0, redshifts, magnitudes, error_magnitudes)

    with Pool() as pool:
        for i, j, L in pool.starmap(marginalized_likelihood, likelihood_args):
            MATRIX[i, j] = L

    # for args in likelihood_args:
    #     i, j, L = new_likelihood(*args)
    #     MATRIX[i, j] = L

    return MATRIX.T


def find_best_fit_values(LIST_Omega_m0, LIST_Omega_Lambda0, MATRIX_likelihood):
    # FIXME: There are some NaN's in the matrix...
    return np.unravel_index(np.nanargmax(MATRIX_likelihood), MATRIX_likelihood.shape)


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



    # === Computation of marginalized likelihood, finding best values for Omega_m0 and Omega_Lambda ===
    # =================================================================================================
    
    # --- define variables ---
    LIST_Omega_m0 = np.arange(0.0, 1.0, 0.005)
    LIST_Omega_Lambda0 = np.arange(0.0, 1.2, 0.005)

    # --- compute marginalized likelihood for every value in LIST_Omega_m0 and LIST_Omega_Lambda0 ---
    MATRIX_like = MATRIX_likelihood(LIST_Omega_m0, LIST_Omega_Lambda0, redshifts, magnitudes, error_magnitudes)
    
    # --- find index of the values for Omega_m0 and Omega_Lambda0 at which the marginalized likelihood has its maximum ---
    Omega_Lambda0_index, Omega_m0_index = find_best_fit_values(LIST_Omega_m0, LIST_Omega_Lambda0, MATRIX_like)
    Omega_m0_best = LIST_Omega_m0[Omega_m0_index]
    Omega_Lambda0_best = LIST_Omega_Lambda0[Omega_Lambda0_index]
    
    # --- print best Omega_m0_best and Omega_Lambda0_best ---
    print("================================")
    print("Values for maximum likelihood: ")
    print("Omega_m0 = ", Omega_m0_best)
    print("Omega_Lambda0 = ", Omega_Lambda0_best)
    print("================================")
    # =================================================================================================




    # === Plot absolute magnitude vs. likelihood with Omega_m0_best and Omega_Lambda0_best ===
    # ========================================================================================
    # To estimate a range in which the absolute_magnitude can be marginalized, we consider the absolute_magnitude vs. likelihood
    
    # --- define variables ---
    d_L = distances(redshifts, Omega_m0_best, Omega_Lambda0_best)
    LIST_absolute_magnitude = np.arange(15.7505, 15.805, 0.0001)
    
    # --- compute likelihood ---
    LIST_likelihood = []
    for i in range(len(LIST_absolute_magnitude)):
        absolute_magnitude = LIST_absolute_magnitude[i]
        L = likelihood(absolute_magnitude, d_L, magnitudes, error_magnitudes)
        LIST_likelihood.append(L)

    # --- fig ---
    fig = plt.figure()
   
    # --- plot absolute magnitude vs. likelihood ---
    plt.plot(LIST_absolute_magnitude, LIST_likelihood)
    plt.xlabel('absolute magnitude $M$')
    plt.ylabel('likelihood $L({0:.2f}, {1:.2f}, M)$'.format(Omega_m0_best, Omega_Lambda0_best))
    plt.suptitle('$\\texttt{chi_square_likelihood.py}$', fontsize=20)
    plt.title('absolute magnitude $M$ vs. likelihood $L({0:.2f}, {1:.2f}, M)$'.format(Omega_m0_best, Omega_Lambda0_best))
    plt.grid(True)
    plt.show()

    # --- save figure ---
    # fig.savefig('../thesis/figures/plots/EPS/[chi_square_likelihood]_absolute_magnitude_vs_likelihood.eps', format = 'eps', bbox_inches = 'tight')
    fig.savefig('../thesis/figures/plots/PNG/[chi_square_likelihood]_absolute_magnitude_vs_likelihood.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    # fig.savefig('../thesis/figures/plots/PDF/[chi_square_likelihood]_absolute_magnitude_vs_likelihood.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/[chi_square_likelihood]_absolute_magnitude_vs_likelihood.tex')
    # ========================================================================================



    # === Computation of Omega_m0 vs. marginalized likelihood with summed Omega_Lambda0 and vice versa ===
    # ====================================================================================================
    
    # --- compute sum_Omega_X0 --- 
    sum_Omega_m0 = np.nansum(MATRIX_like, axis=0)
    sum_Omega_Lambda0 = np.nansum(MATRIX_like, axis=1)
    # COMMENT FOR sum_X0 = np.nansum(MATRIX_like, axis=0)
    # for i in range(len(LIST_Omega_X0)):
    #    s = 0.0
    #    for j in range(len(LIST_Omega_Y0)):
    #        s += MATRIX_like[i, j]
    #    sum_X0[i] = s

    # --- parameter guess for gauss fits p0_guess_X0 = [mu, sigma, y0] ---
    p0_guess_m0 = [0.25, 0.01, 1.0e-124]
    p0_guess_Lambda0 = [0.7, 1.0, 1.0e-124]
    
    # --- calculate parameters by using scipy optimize with defined gauss_curve ---
    popt_m0, pcov_m0 = opt.curve_fit(gauss_curve, LIST_Omega_m0, sum_Omega_m0, p0_guess_m0)
    popt_Lambda0, pcov_Lambda0 = opt.curve_fit(gauss_curve, LIST_Omega_Lambda0, sum_Omega_Lambda0, p0_guess_Lambda0)
    
    mu_m0, sigma_m0, y0_m0 = popt_m0
    mu_Lambda0, sigma_Lambda0, y0_Lambda0 = popt_Lambda0
    
    # --- print parameters for gauss fit ---
    print("=== Parameters for gauss fit ===")
    print("================================")
    print("mu_m0, sigma_m0, y0_m0 = ",  popt_m0)
    print("mu_Lambda0, sigma_Lambda0, y0_Lambda0 = ", popt_Lambda0)
    print("================================")
   
    # --- compute fitted gauss curves ---
    sum_Omega_m0_gauss_fit = gauss_curve(LIST_Omega_m0, *popt_m0)
    sum_Omega_Lambda0_gauss_fit = gauss_curve(LIST_Omega_Lambda0, *popt_Lambda0)

    # --- fig ---
    fig = plt.figure()
   
    # --- plot Omega_m0 vs. marginalized likelihood for summed Omega_Lambda0 with gauss fit ---
    plt.plot(LIST_Omega_m0, sum_Omega_m0, label='data')
    plt.plot(LIST_Omega_m0, sum_Omega_m0_gauss_fit, linestyle='--', color='tab:orange', label='fit')
    plt.xlabel('$\Omega_{m,0}$')
    plt.ylabel('$L(\Omega_{m,0}, \sum \Omega_{\Lambda,0})$')
    plt.suptitle('$\\texttt{chi_square_likelihood.py}$', fontsize=20)
    plt.title('fit values: ($\mu_{{m,0}}, \sigma_{{m,0}}) = ({0:.5f},{1:.5f})$'.format(*popt_m0))
    plt.grid(True) 
    plt.show()
    
    # --- save fig --- 
    # fig.savefig('../thesis/figures/plots/EPS/[chi_square_likelihood]_Omega_m0_vs_likelihood_summed_Omega_Lambda0.eps', format = 'eps', bbox_inches = 'tight')
    fig.savefig('../thesis/figures/plots/PNG/[chi_square_likelihood]_Omega_m0_vs_likelihood_summed_Omega_Lambda0.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    # fig.savefig('../thesis/figures/plots/PDF/[chi_square_likelihood]_Omega_m0_vs_likelihood_summed_Omega_Lambda0.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/[chi_square_likelihood]_Omega_m0_vs_likelihood_summed_Omega_Lambda0.tex')
    
    # --- fig ---
    fig = plt.figure()
    
    # --- plot Omega_Lambda0 vs. marginalized likelihood for summed Omega_m0 with gauss fit ---
    plt.plot(LIST_Omega_Lambda0, sum_Omega_Lambda0, label='data')
    plt.plot(LIST_Omega_Lambda0, sum_Omega_Lambda0_gauss_fit, linestyle='--', color='tab:orange', label='fit')
    plt.xlabel('$\Omega_{\Lambda,0}$')
    plt.ylabel('$L(\sum \Omega_{m,0}, \Omega_{\Lambda,0})$')
    plt.suptitle('$\\texttt{chi_square_likelihood.py}$', fontsize=20)
    plt.title('fit values: ($\mu_{{\Lambda,0}}, \sigma_{{\Lambda,0}}) = ({0:.5f},{1:.5f})$'.format(*popt_Lambda0))
    plt.grid(True) 
    plt.show()

    # --- save fig ---
    # fig.savefig('../thesis/figures/plots/EPS/[chi_square_likelihood]_Omega_Lambda0_vs_likelihood_summed_Omega_m0.eps', format = 'eps', bbox_inches = 'tight')
    fig.savefig('../thesis/figures/plots/PNG/[chi_square_likelihood]_Omega_Lambda0_vs_likelihood_summed_Omega_m0.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    # fig.savefig('../thesis/figures/plots/PDF/[chi_square_likelihood]_Omega_Lambda0_vs_likelihood_summed_Omega_m0.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/[chi_square_likelihood]_Omega_Lambda0_vs_likelihood_summed_Omega_m0.tex')
    # ================================================================================================




    # === Computation of Omega_m0 vs. marginalized likelihood at Omega_Lambda0_best and vice versa ===
    # ================================================================================================
    
    # --- create list of Omega_m0 for which marginalized likelihood is at Omega_Lambda0_best ---
    LIST_Omega_m0_at_Omega_Lambda0_index = MATRIX_like[Omega_Lambda0_index, :]
    # --- create list of Omega_Lambda0 for which marginalized likelihood is at Omega_m0_best ---
    LIST_Omega_Lambda0_at_Omega_m0_index = MATRIX_like[:, Omega_m0_index]
  
    # --- parameter guess for gauss fits p0_guess_X0 = [mu, sigma, y0] ---
    p0_guess_m0 = [0.25, 0.01, 1.0e-124]
    p0_guess_Lambda0 = [0.7, 1.0, 1.0e-124]
    
    # --- calculate parameters by using scipy optimize with defined gauss_curve ---
    popt_m0, pcov_m0 = opt.curve_fit(gauss_curve, LIST_Omega_m0, LIST_Omega_m0_at_Omega_Lambda0_index, p0_guess_m0)
    popt_Lambda0, pcov_Lambda0 = opt.curve_fit(gauss_curve, LIST_Omega_Lambda0, LIST_Omega_Lambda0_at_Omega_m0_index, p0_guess_Lambda0)
    
    mu_m0, sigma_m0, y0_m0 = popt_m0
    mu_Lambda0, sigma_Lambda0, y0_Lambda0 = popt_Lambda0

    # --- print parameters for gauss fit ---
    print("=== Parameters for gauss fit ===")
    print("================================")
    print("mu_m0, sigma_m0, y0_m0 = ",  popt_m0)
    print("mu_Lambda0, sigma_Lambda0, y0_Lambda0 = ", popt_Lambda0)
   
    # --- compute fitted gauss curves ---
    Omega_m0_gauss_fit = gauss_curve(LIST_Omega_m0, *popt_m0)
    Omega_Lambda0_gauss_fit = gauss_curve(LIST_Omega_Lambda0, *popt_Lambda0)

    # --- fig ---
    fig = plt.figure()
    
    # --- plot Omega_m0 vs. marginalized likelihood of Omega_m0 at Omega_Lambda0_best with gauss fit ---
    plt.plot(LIST_Omega_m0, LIST_Omega_m0_at_Omega_Lambda0_index, label='data')
    plt.plot(LIST_Omega_m0, Omega_m0_gauss_fit, linestyle='--', color='tab:orange', label='fit')
    plt.xlabel('$\Omega_{m,0}$')
    plt.ylabel('$L(\Omega_{{m,0}}, {0:.2f})$'.format(Omega_Lambda0_best))
    plt.suptitle('$\\texttt{chi_square_likelihood.py}$', fontsize=20)
    plt.title('fit values: ($\mu_{{m,0}}, \sigma_{{m,0}}) = ({0:.5f},{1:.5f})$'.format(*popt_m0))
    plt.grid(True) 
    plt.show()
    
    # --- save fig --- 
    # fig.savefig('../thesis/figures/plots/EPS/[chi_square_likelihood]_Omega_m0_vs_likelihood_at_Omega_Lambda0_best.eps', format = 'eps', bbox_inches = 'tight')
    fig.savefig('../thesis/figures/plots/PNG/[chi_square_likelihood]_Omega_m0_vs_likelihood_at_Omega_Lambda0_best.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    # fig.savefig('../thesis/figures/plots/PDF/[chi_square_likelihood]_Omega_m0_vs_likelihood_at_Omega_Lambda0_best.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/[chi_square_likelihood]_Omega_m0_vs_likelihood_at_Omega_Lambda0_best.tex')
    
    # --- fig ---
    fig = plt.figure()
    
    # --- plot Omega_Lambda0 vs. marginalized likelihood of Omega_Lambda0 at Omega_m0_best with gauss fit ---
    plt.plot(LIST_Omega_Lambda0, LIST_Omega_Lambda0_at_Omega_m0_index, label='data')
    plt.plot(LIST_Omega_Lambda0, Omega_Lambda0_gauss_fit, linestyle='--', color='tab:orange', label='fit')
    plt.xlabel('$\Omega_{\Lambda,0}$')
    plt.ylabel('$L({0:.2f}, \Omega_{{\Lambda,0}})$'.format(Omega_m0_best))
    plt.suptitle('$\\texttt{chi_square_likelihood.py}$', fontsize=20)
    plt.title('fit values: ($\mu_{{\Lambda,0}}, \sigma_{{\Lambda,0}}) = ({0:.5f},{1:.5f})$'.format(*popt_Lambda0))
    plt.grid(True) 
    plt.show()

    # --- save fig ---
    # fig.savefig('../thesis/figures/plots/EPS/[chi_square_likelihood]_Omega_Lambda0_vs_likelihood_at_Omega_m0_best.eps', format = 'eps', bbox_inches = 'tight')
    fig.savefig('../thesis/figures/plots/PNG/[chi_square_likelihood]_Omega_Lambda0_vs_likelihood_at_Omega_m0_best.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    # fig.savefig('../thesis/figures/plots/PDF/[chi_square_likelihood]_Omega_Lambda0_vs_likelihood_at_Omega_m0_best.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/[chi_square_likelihood]_Omega_Lambda0_vs_likelihood_at_Omega_m0_best.tex')
    # ================================================================================================
    


    # === Compute Omega_m0 vs. Omega_Lambda0 vs. marginalized likelihood ===
    # ======================================================================
    
    # --- compute likelihood ---
    Z = np.array(MATRIX_like)
    X, Y = np.meshgrid(LIST_Omega_m0, LIST_Omega_Lambda0)
    
    # # --- compute sigma_total ---
    # sigma_total = np.sqrt(sigma_m0*sigma_m0 + sigma_Lambda0*sigma_Lambda0)
    # print("sigma_total =", sigma_total)

    # print('********** TOTAL COMPUTATION TIME: %.3f seconds **********'%(time.perf_counter() - START_TOTAL_TIME))
    
    # --- plot 3D ---
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_wireframe(X, Y, Z, edgecolor='blue', alpha=0.3)
    # ax.contourf(X, Y, Z, levels=[k*0.1*sigma_total for k in range(1,30)])
    # ax.contour(X, Y, Z)
    # ax.contourf(X, Y, Z, zdir='x', offset=0, cmap='coolwarm')
    # ax.contourf(X, Y, Z, zdir='y', offset=0, cmap='coolwarm')
    ax.contourf(X, Y, Z, zdir='z', offset=0, cmap='coolwarm')
    ax.set(xlabel='$\Omega_{m,0}$', ylabel='$\Omega_{\Lambda,0}$', zlabel='$L(\Omega_{m,0}, \Omega_{\Lambda,0})$')
    plt.suptitle('$\\texttt{chi_square_likelihood.py}$', fontsize=20)
    plt.title('best fit values: $(\Omega_{m,0}, \Omega_{\Lambda,0}) = (%.2f, %.2f)$'%(Omega_m0_best, Omega_Lambda0_best))
    plt.show()

    # --- save fig ---
    # fig.savefig('../thesis/figures/plots/EPS/[chi_square_likelihood]_Omega_m0_vs_Omega_Lambda0_vs_likelihood.eps', format = 'eps', bbox_inches = 'tight')
    fig.savefig('../thesis/figures/plots/PNG/[chi_square_likelihood]_Omega_m0_vs_Omega_Lambda0_vs_likelihood.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    # fig.savefig('../thesis/figures/plots/PDF/[chi_square_likelihood]_Omega_m0_vs_Omega_Lambda0_vs_likelihood.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/[chi_square_likelihood]_Omega_m0_vs_Omega_Lambda0_vs_likelihood.tex')
    
    # --- plot 2D contour ---
    fig, ax = plt.subplots()
    # ax = plt.axes(projection='2d')

    # ax.plot_wireframe(X, Y, Z, edgecolor='blue', alpha=0.3)
    # ax.contourf(X, Y, Z, levels=[k*0.1*sigma_total for k in range(1,30)])
    # ax.contour(X, Y, Z)
    # ax.contourf(X, Y, Z, zdir='x', offset=0, cmap='coolwarm')
    # ax.contourf(X, Y, Z, zdir='y', offset=0, cmap='coolwarm')
    ax.contourf(X, Y, Z, cmap='coolwarm')
    # ax.set(xlabel='$\Omega_{m,0}$', ylabel='$\Omega_{\Lambda,0}$', zlabel='$L(\Omega_{m,0}, \Omega_{\Lambda,0})$')
    plt.xlabel('$\Omega_{m,0}$') 
    plt.ylabel('$\Omega_{\Lambda,0}$')
    plt.suptitle('$\\texttt{chi_square_likelihood.py}$', fontsize=20)
    plt.title('best fit values: $(\Omega_{m,0}, \Omega_{\Lambda,0}) = (%.2f, %.2f)$'%(Omega_m0_best, Omega_Lambda0_best))
    plt.grid(True)
    plt.show()

    # --- save fig ---
    # fig.savefig('../thesis/figures/plots/EPS/[chi_square_likelihood]_Omega_m0_vs_Omega_Lambda0.eps', format = 'eps', bbox_inches = 'tight')
    fig.savefig('../thesis/figures/plots/PNG/[chi_square_likelihood]_Omega_m0_vs_Omega_Lambda0.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    # fig.savefig('../thesis/figures/plots/PDF/[chi_square_likelihood]_Omega_m0_vs_Omega_Lambda0.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/[chi_square_likelihood]_Omega_m0_vs_Omega_Lambda0.tex')
    # ======================================================================

if __name__ == "__main__":
    main()
