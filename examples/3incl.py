'''This example simulates flow and transport with three inclusions. '''
from Inclusions import *

run_simulation(Lx=3., Ny=50,
                   pack='sqr',n_incl_y=3, Kfactor=0.1,
                   target_incl_area=0.5, radius=None,
                   bcc='flow', isPeriodic=True, integrateInTime=False,
                   tmax=9000.0, dt=None, Npart=int(1e6),
                   plotPerm=False, plotFlow=False,
                   plotTpt=False, plotBTC=False,
                   filename='3incl', doPost=True,
                   doVelPost=True)
