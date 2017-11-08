from Inclusions import *

run_simulation(Lx=1., Ny=100,
                   pack='rnd',n_incl_y=3, Kfactor=0.01,
                   bcc='head',isPeriodic=True, integrateInTime=True,
                   tmax=1.0, dt=None, Npart=5000,
                   plotPerm=True, plotFlow=True,
                   plotTpt=True, plotBTC=True,
                   doPost=True)

#pdb.set_trace()
input("Press enter to continue...")
