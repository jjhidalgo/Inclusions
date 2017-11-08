from Inclusions import *

run_simulation(Lx=1., Ny=100,
                   pack='tri',n_incl_y=3, Kfactor=0.1,
                   bcc='flow', isPeriodic=True, integrateInTime=True,
                   tmax=6., dt=None, Npart=10000,
                   plotPerm=True, plotFlow=True,
                   plotTpt=True, plotBTC=True,
                   doPost=True)
#pdb.set_trace()
input("Press enter to continue...")
