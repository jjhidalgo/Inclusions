from Inclusions import *

run_simulation(Lx=1., Ny=100,
                   pack='tri',n_incl_y=1, Kfactor=0.001,
                   bcc='head', isPeriodic=True, integrateInTime=True,
                   tmax=700.0, dt=None, Npart=int(1e4),
                   plotPerm=True, plotFlow=True,
                   plotTpt=False, plotBTC=True,
                   doPost=True)
#pdb.set_trace()
#input("Press enter to continue...")
