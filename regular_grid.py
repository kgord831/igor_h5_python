import numpy as np
from scipy.interpolate import griddata

dset = np.load('data.npy')[:, :, :]
kx = np.load('kx.npy')[:, :, :]
ky = np.load('ky.npy')[:, :, :]
en = np.load('energy.npy')[:, :, :]
en_axis = en[0, :, 0]

kx_min = kx.min()
ky_min = ky.min()
kx_max = kx.max()
ky_max = ky.max()
e_min = en_axis.min()
de = en_axis[-1] - en_axis[0]
dkx = np.abs(np.mean(np.diff(kx)))
dky = dkx#np.abs(np.mean(np.diff(ky)))
Nkx = int((kx_max - kx_min)/dkx)
Nky = int((ky_max - ky_min)/dky)
new_kx = kx_min + dkx*np.arange(Nkx)
new_ky = ky_min + dky*np.arange(Nky)
kx_mesh, ky_mesh = np.meshgrid(new_kx, new_ky, indexing='ij')

coords = np.array([kx[:, 0, :].flatten(), ky[:, 0, :].flatten()])
coords = coords.T
new_coords = np.array([kx_mesh.flatten(), ky_mesh.flatten()])
new_coords = new_coords.T
dset_regular = griddata(coords, dset[:, 345, :].flatten(), new_coords, method='linear')

np.save('data_grid', dset_regular.reshape(kx_mesh.shape))
np.save('coords', np.array([[kx_min, dkx], [ky_min, dky], [e_min, de]]))

# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.pcolormesh(kx[:, 345, :], ky[:, 345, :], dset[:, 345, :])
# plt.figure(2)
# plt.pcolormesh(kx_mesh, ky_mesh, dset_regular.reshape(kx_mesh.shape))
# plt.show()