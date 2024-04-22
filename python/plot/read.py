import h5py
import numpy as np

from scipy.special import eval_legendre, roots_legendre
from legendre import pl_eval_2D, pl_project_2D
from plot import plot_scalar, get_cmap_from_proplot


#--------------------#
#   Initialisation   #
#--------------------#
# File reading
f = h5py.File('model2d.h5', 'r')

# Constants
I = f['/star/'].attrs['nr']        # number of radial points
D = f['/star/'].attrs['ndomains']  # number of domains
J = f['/star/'].attrs['nth'] * 2   # number of angular points

# Variables
r = f['/star/r'][:].T            # radial coordinate
z = f['/star/z'][0, :]           # zeta grid
c = np.cos(f['/star/th'][:, 0])  # values of cos(theta)
p = f['/star/p'][:].T            # pressure
w = f['/star/w'][:].T            # angular velocity
g = f['/star/G'][:].T            # meridional stream function
n2 = f['/star/N2'][:].T          # buoyancy frequency squared

# Restoring the entiere variables
r = np.concatenate((r[:, ::-1], r), axis=1)
c = np.concatenate((-c[::-1]  , c), axis=0)
p = np.concatenate((p[:, ::-1], p), axis=1)
w = np.concatenate((w[:, ::-1], w), axis=1)
g = np.concatenate((g[:, ::-1], g), axis=1)
n2 = np.concatenate((n2[:, ::-1], n2), axis=1)

# End file reading
f.close()

#--------------------#
#        Plot        #
#--------------------#
L = 16
size = 24
cmap1 = get_cmap_from_proplot('balance')
cmap2 = get_cmap_from_proplot('br')

# Angular velocity
disc = np.min(np.arange(I)[n2[:, J//2] > 1e-10]).reshape(1, )

# Stream function
s = (1 - c**2)**0.5
psi = np.where(n2 > 0.0, g * (r * s), 0.0)
f = psi
eps, levels = 0.1, 50
levels = np.concatenate((
    -f.max() * np.linspace(1.0, eps, levels//2),
    +f.max() * np.linspace(eps, 1.0, levels//2)
))

# Plot
plot_scalar(
    r/r.max(), w, L, cmap=cmap1, size=size, levels=50, label=r"$\Omega/\Omega_K$"
)
plot_scalar(
    r/r.max(), f, L, cmap=cmap2, size=size, levels=levels, 
    div=True, potential=True, lw=1.5, disc=disc, disc_color='k',
    label=r"$\Psi$"
)


