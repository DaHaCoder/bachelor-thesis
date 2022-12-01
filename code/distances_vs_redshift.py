### matplotlib package -- https://matplotlib.org/stable/index.html ###
from matplotlib import pyplot as plt    #   for plotting 

### numpy package -- https://numpy.org/doc/stable/ ###
import numpy as np                      #   for general scientific computation

# ### scipy package -- https://docs.scipy.org/doc/scipy/index.html ###
# from scipy import constants as const    #   for physical constants -- https://docs.scipy.org/doc/scipy/reference/constants.html 
from scipy.integrate import quad          #   for integration -- https://docs.scipy.org/doc/scipy/tutorial/integrate.html
# from scipy import optimize as opt       #   for optimization and fit -- https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
# from scipy import special as sp         #   for special mathematical functions -- https://docs.scipy.org/doc/scipy/reference/tutorial/special.html

plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = r'''
# '''


def integrand(x, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0):
    if x == 0.0:
        return 0.0
    return 1.0/np.sqrt(Omega_r0*(1.0+x)**(4.0) + Omega_m0*(1.0+x)**(3.0) + Omega_K0*(1.0+x)**(2.0) + Omega_Lambda0)


def integral(z, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0):
    # D_C/D_H = Integrate[1/E(z'), {z',0,z}]
    return quad(integrand, 0.0, z, args=(Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0))[0]


def distances(z, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0, hubble_distance):
    I = np.array([integral(zi, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0) for zi in z])

    if Omega_K0 > 0.0:
        proper_motion_distance = hubble_distance*1.0/np.sqrt(Omega_K0)*np.sinh(np.sqrt(Omega_K0)*I)

    elif Omega_K0 == 0.0:
        proper_motion_distance = hubble_distance*I

    elif Omega_K0 < 0.0:
        proper_motion_distance = hubble_distance*1.0/np.sqrt(abs(Omega_K0))*np.sin(np.sqrt(abs(Omega_K0))*I)

    angular_diameter_distance = 1/(1.0 + z)*proper_motion_distance
    luminosity_distance = (1.0 + z)*proper_motion_distance

    return proper_motion_distance, angular_diameter_distance, luminosity_distance 


def main():    
    z_min = 0.01 
    z_max = 100
    z = np.linspace(z_min, z_max, 10000)
    
    Omega_r0 = 0.0
    Omega_m0 = 0.3
    Omega_Lambda0 = 0.7
    # Omega_K0 = 0.0
    Omega_K0 = 1.0 - Omega_r0 - Omega_m0 - Omega_Lambda0

    c = 299792.458
    h = 0.7
    H_0 = h*100 
    d_H = c/H_0

    d_M = distances(z, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0, d_H)[0]
    d_A = distances(z, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0, d_H)[1]
    d_L = distances(z, Omega_r0, Omega_m0, Omega_K0, Omega_Lambda0, d_H)[2]

    fig, ax = plt.subplots()
    
    plt.plot(z, d_M, color = 'red', label = 'proper motion distance $d_{M}$')
    plt.plot(z, d_A, color = 'blue', label = 'angular diameter distance $d_{A}$')
    plt.plot(z, d_L, color = 'green', label = 'luminosity distance $d_{L}$')

    ax.set_xscale('log', base = 10)
    ax.set_yscale('log', base = 10)

    plt.title('Cosmological distances $d_{i}$ vs. redshift $z$ for $\\Omega_{r,0} = %.1f$, $\\Omega_{m,0} = %.1f$, $\\Omega_{\\Lambda,0} = %.1f$'%(Omega_r0, Omega_m0, Omega_Lambda0))
    plt.xlabel('redshift $z$')
    plt.ylabel('$d_{i}/d_{H}$')
    plt.legend(loc = 'upper left')
    plt.grid(True)

    plt.show()

    # fig.savefig('../thesis/figures/plots/EPS/distances_vs_redshift.eps', format = 'eps', bbox_inches = 'tight')
    fig.savefig('../thesis/figures/plots/PNG/distances_vs_redshift.png', format = 'png', bbox_inches = 'tight', dpi = 250)
    # fig.savefig('../thesis/figures/plots/PDF/distances_vs_redshift.pdf', format = 'pdf', bbox_inches = 'tight')
    # tikzplotlib.save('../thesis/figures/tikz/distances_vs_redshift.tex')


if __name__ == "__main__":
    main()
