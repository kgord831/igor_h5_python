import numpy as np
from scipy.interpolate import griddata

dset = np.load('data.npy')[::2, ::2, :]
kx = np.load('kx.npy')[::2, ::2, :]
ky = np.load('ky.npy')[::2, ::2, :]
en = np.load('energy.npy')[::2, ::2, :]
en_axis = en[0, :, 0]

kx_min = kx[:, 0, :].min()
ky_min = ky[:, 0, :].min()
kx_max = kx[:, 0, :].max()
ky_max = ky[:, 0, :].max()
e_min = en_axis.min()
de = en_axis[-1] - en_axis[0]
dkx = np.abs(np.mean(np.diff(kx)))
dky = dkx#np.abs(np.mean(np.diff(ky)))
Nkx = int((kx_max - kx_min)/dkx)
Nky = int((ky_max - ky_min)/dky)
new_kx = kx_min + dkx*np.arange(Nkx)
new_ky = ky_min + dky*np.arange(Nky)
kx_mesh, ky_mesh = np.meshgrid(new_kx, new_ky, indexing='ij')

dset_regular = np.empty((Nkx, len(en_axis), Nky))
for i in range(len(en_axis)):
    print(i)
    coords = np.array([kx[:, i, :].ravel(), ky[:, i, :].ravel()])
    coords = coords.T
    new_coords = np.array([kx_mesh.ravel(), ky_mesh.ravel()])
    new_coords = new_coords.T
    dset_regular[:, i, :] = griddata(coords, dset[:, i, :].ravel(), new_coords, method='linear').reshape((Nkx, Nky))

np.save('data_grid', dset_regular)
np.save('coords', np.array([[kx_min, dkx], [e_min, de], [ky_min, dky]]))

# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.pcolormesh(kx[:, 345, :], ky[:, 345, :], dset[:, 345, :])
# plt.figure(2)
# plt.pcolormesh(kx_mesh, ky_mesh, dset_regular.reshape(kx_mesh.shape))
# plt.show()