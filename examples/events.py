'''This example simulates flow and transport with three inclusions. '''
from Inclusions import *

run_simulation(Lx=3., Ny=100,
                   pack='sqr',n_incl_y=1, Kfactor=0.1,
                   target_incl_area=0.5, radius=None,
                   bcc='flow', isPeriodic=True, integrateInTime=True,
                   transportMethod='time',
                   tmax=9000.0, dt=None, Npart=int(2e1),
                   plotPerm=False, plotFlow=False,
                   plotTpt=True, plotBTC=False,
                   filename='zz', doPost=True,
                   doVelPost=False)
