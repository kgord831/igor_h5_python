import h5py
import numpy as np
from scipy import ndimage

def load_wave_from_h5(file_name, wave_name):
    f = h5py.File('map.h5', 'r')
    dset = f[wave_name]
    rank = len(dset.shape)
    dim_sizes = dset.shape
    dim_scales = []
    for i in range(rank):
        dim_scales.append(np.arange(dim_sizes[i])*dset.attrs['IGORWaveScaling'][i + 1,0] + dset.attrs['IGORWaveScaling'][i + 1,1])
    return tuple([dset] + dim_scales)

wave_name = 'HoBi_f00000'

dset, phi_axis, en_axis, theta_axis = load_wave_from_h5('map.h5', wave_name)
en_axis = 92 + en_axis - 4.3
print(dset.shape)
print(phi_axis.shape, en_axis.shape, theta_axis.shape)

# Phi is along the list
# Theta is perpendicular to the slit
phi_offset = 1.313
theta_offset = -2.4
# Offset the axes, convert to radians
phi_axis = (phi_axis - (phi_offset))*np.pi/180
theta_axis = (theta_axis - (theta_offset))*np.pi/180
# Construct the coordinate grid
phi_axis, en_axis, theta_axis = np.meshgrid(phi_axis, en_axis, theta_axis, indexing='ij')
# Calculate the k points
# 0.512 = Sqrt(2*electron mass)/hbar (in units of A^-1/sqrt(eV))
kperp = 0.512*np.sqrt(en_axis)*np.sin(theta_axis)
kpar = 0.512*np.sqrt(en_axis)*np.cos(theta_axis)*np.sin(phi_axis)
# Rotate the azimuth
azimuth = -30*np.pi/180
kperp_rot = np.cos(azimuth)*kperp - np.sin(azimuth)*kpar
kpar_rot = np.sin(azimuth)*kperp + np.cos(azimuth)*kpar
np.save('data', dset)
np.save('kx', kpar_rot)
np.save('ky', kperp_rot)
np.save('energy', en_axis)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
# Plot the ith energy isosurface
#i = 432
i = 230
imin = i - 5
imax = i + 5
for i in [432, 345, 190]:
    imin = i - 5
    imax = i + 5
    fig, ax = plt.subplots(1)
    dset_sum = np.sum(dset[:,imin:imax,:], axis=1)
    ax.pcolormesh(kperp_rot[:, i, :], kpar_rot[:, i, :], dset_sum)
    t = 0.99597
    bz1 = np.array([[-1,-0.5], [-1, 0.5], [-0.5, 1], [0.5, 1], [1, 0.5], [1, -0.5], [0.5, -1], [-0.5, -1], [-1, -.5]])
    bz2 = np.array([[0.5, 1], [1, 1.5], [1.5, 1], [1, 0.5]])
    ax.plot(t*bz1[:, 0], t*bz1[:, 1], linewidth=2, color='r', ls='--')
    ax.plot(t*bz2[:, 0], t*bz2[:, 1], linewidth=2, color='b', ls='--')
    if i == 190:
        ax.set_title('Binding Energy = -1.95 eV')
    elif i == 345:
        ax.set_title('Binding Energy = -0.7 eV')
    elif i == 432:
        ax.set_title('Fermi Surface')
    ax.set_xlabel(r'$k_x$ $(\AA^{-1})$')
    ax.set_ylabel(r'$k_y$ $(\AA^{-1})$')
    ax.text(0, 0, '$\Gamma$', color='yellow', fontsize=14)
    ax.text(0, t*1, '$X$', color='yellow', fontsize=14)
    ax.text(0.5*t, 1*t, '$W$', color='yellow', fontsize=14)
    ax.text(0.75*t, 0.75*t, '$K$', color='yellow', fontsize=14)
    #bz = patches.RegularPolygon((0, 0), 8, 1.11352816733, 22.5*np.pi/180, linewidth=2, edgecolor='r', ls='--', facecolor='none')
    #rect = patches.Rectangle((-np.pi/6.2291, -np.pi/6.2291), 2*np.pi/6.2291, 2*np.pi/6.2291, linewidth=2, edgecolor='r', ls='--', facecolor='none')
    #ax.add_patch(bz)
plt.show()
