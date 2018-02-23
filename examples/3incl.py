'''This example simulates flow and transport with three inclusions. '''
from Inclusions import *

run_simulation(Lx=3., Ny=None,
                   pack='tri',n_incl_y=3, Kfactor=0.1,
                   target_incl_area=0.5, radius=None,
                   bcc='flow', isPeriodic=False, integrateInTime=False,
                   tmax=9000.0, dt=None, Npart=int(1e5),
                   plotPerm=True, plotFlow=False,
                   plotTpt=False, plotBTC=False,
                   filename='3incl2', doPost=True)
