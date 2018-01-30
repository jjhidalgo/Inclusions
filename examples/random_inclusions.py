'''This example simulates flow and transport in a domain with
 a random arangement of inclusions.'''
from Inclusions import *
run_simulation(Lx=1., Ny=400,
                   pack='rnd',n_incl_y=10, Kfactor=0.1,
                   target_incl_area=0.4, radius=None,
                   bcc='flow', isPeriodic=True, integrateInTime=False,
                   tmax=None, dt=None, Npart=int(1e4),
                   plotPerm=True, plotFlow=False,
                   plotTpt=False, plotBTC=False,
                   filename='test-rnd', doPost=True)
