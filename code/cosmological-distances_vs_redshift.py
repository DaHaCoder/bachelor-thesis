### matplotlib package -- https://matplotlib.org/stable/index.html ###
from matplotlib import pyplot as plt                                 #   for plotting 

### numpy package -- https://numpy.org/doc/stable/ ###
import numpy as np                                   #   for general scientific computation

### scipy package -- https://docs.scipy.org/doc/scipy/index.html ###
from scipy.integrate import quad                                   #   for integration -- https://docs.scipy.org/doc/scipy/tutorial/integrate.html

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'''
\usepackage{physics}
'''


def integrand(x, Omega_r0, Omega_m0, Omega_Lambda0):
    if x == 0.0:
        return 1.0
    Omega_K0 = 1.0 - Omega_r0 - Omega_m0 - Omega_Lambda0
    
    # define one_x := 1.0 + x and replace higher powers of it by recursive multiplication of one_x, since '**'-operator may be slow
    one_x = 1.0 + x
    one_x2 = one_x * one_x
    one_x3 = one_x2 * one_x
    one_x4 = one_x2 * one_x2
    return 1.0/np.sqrt(Omega_r0*one_x4 + Omega_m0*one_x3 + Omega_K0*one_x2 + Omega_Lambda0)


def integral(z, Omega_r0, Omega_m0, Omega_Lambda0):
    # d_C/d_H = Integrate[1/E(z'), {z',0,z}]
    return quad(integrand, 0.0, z, args=(Omega_r0, Omega_m0, Omega_Lambda0))[0]


def distances(z, Omega_r0, Omega_m0, Omega_Lambda0):
    # Cosmological Parameters
    # =======================
    c = 299792.458
    h = 0.6766               # value obtained by Planck Collaboration, "Planck 2018 results", Table 7 -- https://doi.org/10.1051/0004-6361/201833880
    H_0 = h*100.0
    d_H = c/H_0
    # =======================
    
    I = np.array([integral(zi, Omega_r0, Omega_m0, Omega_Lambda0) for zi in z])

    Omega_K0 = 1.0 - Omega_r0 - Omega_m0 - Omega_Lambda0
    if Omega_K0 > 0.0:
        transverse_comoving_distance = d_H * 1.0/np.sqrt(Omega_K0) * np.sinh(np.sqrt(Omega_K0) * I)

    elif Omega_K0 == 0.0:
        transverse_comoving_distance = d_H * I

    elif Omega_K0 < 0.0:
        transverse_comoving_distance = d_H * 1.0/np.sqrt(abs(Omega_K0)) * np.sin(np.sqrt(abs(Omega_K0)) * I)

    angular_diameter_distance = 1/(1.0 + z) * transverse_comoving_distance
    luminosity_distance = (1.0 + z) * transverse_comoving_distance

    return transverse_comoving_distance, angular_diameter_distance, luminosity_distance 


def main():    
    z_min = 0.01 
    z_max = 100
    z = np.linspace(z_min, z_max, 10000)
    
    Omega_r0 = 0.0
    Omega_m0 = 0.3
    Omega_Lambda0 = 0.7

    d_C = distances(z, Omega_r0, Omega_m0, Omega_Lambda0)[0]
    d_A = distances(z, Omega_r0, Omega_m0, Omega_Lambda0)[1]
    d_L = distances(z, Omega_r0, Omega_m0, Omega_Lambda0)[2]

    fig, ax = plt.subplots()
    
    plt.plot(z, d_C, color='red', label='comoving distance $d_{\\text{C}}$')
    plt.plot(z, d_A, color='blue', label='angular diameter distance $d_{\\text{A}}$')
    plt.plot(z, d_L, color='green', label='luminosity distance $d_{\\text{L}}$')

    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)

    # plt.title('Cosmological distances $d_{\\text{{i}}}(z)$ for $(\\Omega_{{\\text{{r}},0}}, \\Omega_{{\\text{{m}},0}}, \\Omega_{{\\Lambda,0}}) = ({0:.1f}, {1:.1f}, {2:.1f})$'.format(Omega_r0, Omega_m0, Omega_Lambda0))
    plt.xlabel('redshift $z$')
    plt.ylabel('cosmological distances $d_{\\text{i}}(z)$')
    plt.legend(loc='upper left')
    plt.grid(True)

    # plt.show()

    fig.savefig('../thesis/figures/plots/EPS/cosmological-distances_vs_redshift.eps', format='eps', bbox_inches='tight')
    fig.savefig('../thesis/figures/plots/PNG/cosmological-distances_vs_redshift.png', format='png', bbox_inches='tight', dpi = 400)
    fig.savefig('../thesis/figures/plots/PDF/cosmological-distances_vs_redshift.pdf', format='pdf', bbox_inches='tight')
    # tikzplotlib.save('../thesis/figures/tikz/cosmological-distances_vs_redshift.tex')


if __name__ == "__main__":
    main()
