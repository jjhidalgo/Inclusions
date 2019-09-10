'''This example simulates flow and transport with three inclusions. '''
from Inclusions import *

run_simulation(Lx=3., Ny=400,
                   pack='sqr',n_incl_y=3, Kfactor=0.1,
                   target_incl_area=0.5, radius=None,
                   bcc='flow', isPeriodic=True, integrateInTime=False,
                   transportMethod='pollock',
                   tmax=9000.0, dt=None, Npart=int(1e4),
                   plotPerm=False, plotFlow=False,
                   plotTpt=False, plotBTC=False,
                   filename='3inclp', doPost=True,
                   doVelPost=False, control_planes=4)
