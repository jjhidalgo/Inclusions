'''This example simulates flow and transport with three inclusions. '''
from Inclusions import *

run_simulation(Lx=1., Ny=None,
                   pack='tri',n_incl_y=2, Kfactor=0.1,
                   target_incl_area=0.5, radius=None,
                   bcc='head', isPeriodic=True, integrateInTime=False,
                   tmax=9000.0, dt=None, Npart=int(1.1e1),
                   plotPerm=False, plotFlow=False,
                   plotTpt=False, plotBTC=False,
                   filename='3incl', doPost=True)
