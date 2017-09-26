from Inclusions import *
#from Kcircles import *
Ly = 1.
Lx = 1.
Ny = 50
Nx = np.int(Ny*Lx)
Kfactor = 0.1
nry = 1
pack = 'tri'
plotKperm = False
kperm = permeability(nry, Lx, Nx, Ny, Kfactor, pack, plotKperm)

bcc = True
plotHead = False
ux, uy = flow(Nx, Ny, Lx, Ly ,kperm, bcc, plotHead)

tmax = 10.
tx = Lx/np.float(Nx)/np.max(ux)
ty = Ly/np.float(Nx)/np.max(np.abs(uy))
dt = np.min([tx ,ty, 1e-3])
print([tx,ty])
#dt = 1e-2

Npart = np.int(1e5)
cbtc, time = transport(Npart, ux, uy, Nx, Ny, Lx, Ly, tmax, dt, True, kperm)

#np.savetxt('K01sqrNy5.dat', np.matrix([time, cbtc]).transpose())
figc = None
axc = None
linc = None
figc, axc, linc = plotXY(time, 1-cbtc, fig=figc, ax=axc, lin=linc)
#pdb.set_trace()
raw_input("Press enter to continue...")
