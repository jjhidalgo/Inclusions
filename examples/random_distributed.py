'''This example simulates flow and transport in a domain with
 a random arangement of inclusions.'''
from Inclusions import *
rr = 0.0718096104723
run_simulation(Lx=2., Ny=200,
                   pack='rnd',n_incl_y=5, Kfactor=[0.1, 1.],Kdist='lognorm',
                   target_incl_area=0.4, radius=[rr/2.,rr],
                   bcc='flow', isPeriodic=True,
                   plotPerm=True, plotFlow=True,
                   calcPsi=True, plotPsi=True,savePsi=False,
                   transportMethod=None,
                   filename='test-size', doPost=False,
                   directSolver=False, overlapTol=0.2)
