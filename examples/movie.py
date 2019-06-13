'''This example simulates flow and transport with one inclusion. '''
from Inclusions import *



run_simulation(Lx=3., Ny=None,
                   pack='rnd',n_incl_y=5, Kfactor=0.1,
                   target_incl_area=0.4, radius=None,
                   bcc='head', isPeriodic=True, integrateInTime=True,
                   transportMethod='time',
                   tmax=9000.0, dt=0.005, Npart=int(1e3),
                   plotPerm=False, plotFlow=False,
                   plotTpt=True, plotBTC=False,
                   filename='1inclt', doPost=False)

