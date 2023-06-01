### matplotlib package -- https://matplotlib.org/stable/index.html ###
from matplotlib import pyplot as plt                                 #   for plotting

### numba package -- https://numba.pydata.org/numba-doc/latest/index.html ###
from numba import njit                                                      #   for faster code compilation ('jit' = just-in-time) -- https://numba.pydata.org/numba-doc/latest/user/5minguide.html?highlight=njit

### numpy package -- https://numpy.org/doc/stable/ ###
import numpy as np                                   #   for general scientific computation

### scipy package -- https://docs.scipy.org/doc/scipy/index.html ###
from scipy.integrate import quad                                   #   for integration -- https://docs.scipy.org/doc/scipy/tutorial/integrate.html
from scipy import optimize as opt                                  #   for optimization and fit -- https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html

plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'''
\usepackage{physics}
'''


# functions in Lambda-CDM-Cosmology #
# ================================= #
@njit
def expansion_function(z, Omega_r0, Omega_m0, Omega_Lambda0):
    # The expansion function E(z) follows from the friedmann equation with
    # H(z) = H_0 np.sqrt(Omega_r0 * (1.0 + z)^4 + Omega_m0 * (1.0 + z)^3 + Omega_K0 * (1.0 + z)^2 + Omega_Lambda0) =: H_0 * E(z)
    Omega_K0 = 1.0 - Omega_r0 - Omega_m0 - Omega_Lambda0
    return np.sqrt(Omega_r0 * np.power(1.0 + z, 4) + Omega_m0 * np.power(1.0 + z, 3) + Omega_K0 * np.power(1.0 + z, 2) + Omega_Lambda0)


@njit
def integrand(z, Omega_r0, Omega_m0, Omega_Lambda0):
    if z == 0.0:
        return 1.0
    E = expansion_function(z, Omega_r0, Omega_m0, Omega_Lambda0)
    return 1.0/E


def integral(z, Omega_r0, Omega_m0, Omega_Lambda0):
    # d_C/d_H = Integrate[1/E(z'), {z', 0, z}]
    return quad(integrand, 0.0, z, args=(Omega_r0, Omega_m0, Omega_Lambda0))[0]


def luminosity_distance(z, Omega_m0, Omega_Lambda0):
    # Cosmological Parameters
    # =======================
    c = 299792.458           # speed of light in vacuum in km/s
    h = 0.6766               # Planck Collaboration 2018, Table 7, Planck+BAO -- https://www.aanda.org/articles/aa/full_html/2020/09/aa33880-18/T7.html  
    H_0 = h*100.0            # hubble constant in km/s per Mpc
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
# ================================= #



# functions in DGP-Cosmology #
# ========================== #
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
    h = 0.6766               # Planck Collaboration 2018, Table 7, Planck+BAO -- https://www.aanda.org/articles/aa/full_html/2020/09/aa33880-18/T7.html  
    H_0 = h*100.0            # hubble constant in km/s per Mpc
    d_H = c/H_0              # hubble distance
    # =======================
    sample_redshifts = np.linspace(0.0, max(z), 1000)
    sample_E = np.array([sol_friedmann(zi, Omega_m0, alpha, mod_friedmann, deriv_mod_friedmann) for zi in sample_redshifts])

    I = np.array([interp_integral(zi, sample_redshifts, sample_E) for zi in z])

    return (1.0 + z) * d_H * I
# ========================== #


@njit
def relative_magnitude(absolute_magnitude, luminosity_distance):
    # luminosity_distance per Mpc, absolute_magnitude is at 10 pc (therefore + 25.0 since 10 pc = 10^(-5.0) Mpc)
    return absolute_magnitude + 5.0 * np.log10(luminosity_distance) + 25.0


def main():    
    # Import data

    DATA_DIR = "../data/SN-data.txt"
    names, redshifts, magnitudes, error_magnitudes = np.loadtxt(DATA_DIR,
                                                                comments="#",
                                                                usecols=(0, 1, 2, 3),
                                                                dtype=np.dtype([("name", str),
                                                                                ("redshift", float),
                                                                                ("magnitude", float),
                                                                                ("sigma", float)]),
                                                                unpack=True)

   
    z_min = min(redshifts)
    z_max = max(redshifts)
    z = np.linspace(z_min, z_max, 1000)

    fig, ax = plt.subplots()   
    
    plt.errorbar(redshifts, magnitudes, yerr=error_magnitudes, fmt='.', elinewidth=1, alpha=0.4, label='Union2.1') 
   
   
    # magnitude vs. redshift for several cosmologies
    # ==============================================
    
    # Only matter (Einstein-de-Sitter universe)
    # -----------------------------------------
    Omega_m0 = 1.0
    Omega_Lambda0 = 0.0

    d_L = luminosity_distance(z, Omega_m0, Omega_Lambda0)
    m = relative_magnitude(0.0, d_L)

    plt.plot(z, m, color='blue', label='$(\\Omega_{{\\text{{m}},0}}, \\Omega_{{\\Lambda,0}}) = ({0:.1f}, {1:.1f})$ (Einstein--de--Sitter)'.format(Omega_m0, Omega_Lambda0))
    # -----------------------------------------
   
    # Lambda-CDM Model
    # ----------------
    Omega_m0 = 0.3
    Omega_Lambda0 = 0.7
    
    d_L = luminosity_distance(z, Omega_m0, Omega_Lambda0)
    m = relative_magnitude(0.0, d_L)
    
    plt.plot(z, m, color='red', label='$(\\Omega_{{\\text{{m}},0}}, \\Omega_{{\\Lambda,0}}) = ({0:.1f}, {1:.1f})$ ($\\Lambda$CDM)'.format(Omega_m0, Omega_Lambda0))
    # ----------------
   
    # # alpha = 0.0
    # # -----------
    # Omega_m0 = 0.3
    # alpha = 0.0
    
    # d_L = DGP_luminosity_distance(z, Omega_m0, alpha)
    # m = relative_magnitude(0.0, d_L)
    
    # plt.plot(z, m, color='green', linestyle='dashdot', label='($\\Omega_{{\\text{{m}},0}}, \\alpha) = ({0:.1f}, {1:.1f})$'.format(Omega_m0, alpha))
    # ---------
    
    # DGP-Model
    # ---------
    Omega_m0 = 0.3
    alpha = 1.0
    
    d_L = DGP_luminosity_distance(z, Omega_m0, alpha)
    m = relative_magnitude(0.0, d_L)
    
    plt.plot(z, m, color='orange', label='$(\\Omega_{{\\text{{m}},0}}, \\alpha) = ({0:.1f}, {1:.1f})$ (DGP)'.format(Omega_m0, alpha))
    # ---------

    # ==============================================

    # plt.title('distance modulus $m - M$ vs. redshift $z$ for Type Ia supernovae in several cosmologies')
    plt.xlabel('redshift $z$')
    plt.ylabel('distance modulus $m - M$')
    plt.legend(loc='lower right')
    plt.grid(True)
   
    # plt.show()

    fig.savefig('../thesis/figures/plots/EPS/redshift-vs-distance-modulus.eps', format='eps', bbox_inches='tight')
    fig.savefig('../thesis/figures/plots/PNG/redshift-vs-distance-modulus.png', format='png', bbox_inches='tight', dpi=250)
    fig.savefig('../thesis/figures/plots/PDF/redshift-vs-distance-modulus.pdf', format='pdf', bbox_inches='tight')
    # tikzplotlib.save('../thesis/figures/tikz/redshift-vs-distance-modulus.tex')

if __name__ == "__main__":
    main()
