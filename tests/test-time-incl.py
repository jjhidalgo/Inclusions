''' Script to test that the residence time of particles in inclusions,
    mobile time, and inclusions visited at a certain time are correctly
    computed when integrating along streamlines.'''
import numpy as np
import scipy.sparse as sp
import Inclusions as II
import ipdb as ipdb


Lx = 1.
Ny = 10.
grid = II.setup_grid(Lx, Ny)
Lx, Ly, Nx, Ny = II.unpack_grid(grid)
dx = Lx/Nx
dy = Ly/Ny

#permeability

kperm = np.ones([Ny, Nx])
incl_ind = np.zeros([Ny, Nx])

x1 = np.arange(dx/2.0, Lx, dx) #cell centers' coordinates
y1 = np.arange(dy/2.0, Ly, dy)
xx, yy = np.meshgrid(x1, y1)

kperm[xx>-1] = 1.
mm1 = (yy>0.5) & (xx>0.25) & (xx<0.5)
kperm[mm1] = 0.1
mm2 = (yy<0.5) & (xx>0.25) & (xx<0.5)
kperm[mm2] = 0.1

incl_ind[xx>-1] = 0
incl_ind[mm1] = 1
incl_ind[mm2] = 2
incl_ind = sp.csr_matrix(incl_ind)

Npart = np.int(3e0)
ux = np.ones([Ny, Nx + 1])
uy = np.zeros([Ny + 1, Nx])
x1 = np.arange(0., Lx + dx, dx) #faces' coordinates
xx, yy = np.meshgrid(x1, y1)
#mm1 = (yy>0.5) & (xx>0.25) & (xx<0.5)
#ux[mm1] = 0.1
tmax = None
ds = 0.01*Lx/Nx
time_ds, t_in_incl_ds = II.transport_ds(grid, incl_ind, Npart, ux, uy,
                                           ds, isPeriodic=False)


tmax = 5.
dt = 0.001
time_dt, t_in_incl_dt = II.transport(grid, incl_ind, Npart, ux, uy,
                                     tmax, dt, isPeriodic=False,
                                     plotit=False, CC=None)
ipdb.set_trace()
#input("Press enter to continue...")
