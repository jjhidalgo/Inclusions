'''This example simulates flow and transport with one inclusion. '''
from Inclusions import *

run_simulation(Lx=1., Ny=None,
                   pack='tri',n_incl_y=1, Kfactor=0.1,
                   target_incl_area=0.5, radius=None,
                   bcc='head', isPeriodic=True, integrateInTime=True,
                   transportMethod='time',
                   tmax=9000.0, dt=None, Npart=int(1e4),
                   plotPerm=False, plotFlow=False,
                   plotTpt=False, plotBTC=False,
                   filename='1inclt', doPost=True)
