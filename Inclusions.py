""" This module provides functions to solve the flow and transport equations
    in porous media at the Darcy scale. The domain is rectangular and the
    permeability field can contain circular inclusions arranged regularly or
    randomly.

    Author: Juan J. Hidalgo, IDAEA-CSIC, Barcelona, Spain.
    Acknowledgements:
        Project MHetScale (FP7-IDEAS-ERC-617511)
            European Research Council

        Project Mec-MAT (CGL2016-80022-R)
            Spanish Ministry of Economy and Competitiveness
"""
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lgsp
import ipdb
verbose = True #False
################
def run_simulation(*, Lx=1., Ny=50,
                   pack='tri', n_incl_y=3, Kfactor=0.1,Kdist='const',
                   target_incl_area=0.5, radius=None,
                   bcc='head', isPeriodic=True, integrateInTime=False,
                   flowMethod='CalcPerm',
                   saveVel=False,
                   calcPsi=False, plotPsi=False,savePsi=False,
                   transportMethod='pollock',
                   tmax=10., dt=None, Npart=100,
                   plotPerm=False, plotFlow=False,
                   plotTpt=False, plotBTC=False,
                   filename=None, doPost=True, doVelPost=False,
                   directSolver=True, tol=1e-10, maxiter=2000,
                   overlapTol=None, control_planes=1, InjSize=1.,
                   reactive=False, Rfactor=0., Rdist='const', Diff=None):
    """ Runs a simulation."""

# Flow methods : CalcPerm, ReadPerm, ReadVel
#    CalcPer: Computespermeability according to given parameters.
#    ReadPerm:  Reads previous permeability geometry.
#               Given Kfactor and Kdist are used instead of the old ones.
#    ReadVel: Reads velicity from file. Flow not solved.

    Kfactor = np.atleast_1d(Kfactor)

    if flowMethod == 'CalcPerm':
        print('calculo')
        grid = setup_grid(Lx, Ny)
        kperm, incl_ind, grid, xcp, Kincl = permeability(grid, n_incl_y, Kfactor,Kdist=Kdist,
                                                  target_incl_area=target_incl_area,
                                                  radius=radius, isPeriodicK=isPeriodic,
                                                  pack=pack, filename=filename,
                                                  plotit=plotPerm, saveit=True,
                                                  overlapTol=overlapTol,
                                                  calcKinfo=True, calcKeff=False,
                                                  calcKhist = False,
                                                  control_planes=control_planes)

    if flowMethod == 'ReadPerm':
        print('Reading permeability from file...')

        grid, circles, Kfactor_old, Kdist_old, Kincl = load_perm('./perm/' + filename)

        if Kdist_old == Kdist and np.array_equal(Kdist_old, Kdist):
            print('Permeability did not change.')
        else:
            print('I use the new Kfactor and Kdist.')
            Kincl = None

        kperm, incl_ind, Kincl = perm_matrix(grid, circles, Kfactor, Kdist=Kdist, Kincl=Kincl)
        save_perm(grid, circles, Kfactor, Kdist, Kincl, filename=filename)
        
        if plotPerm:
            isLogNorm = 'log' in Kdist
            plot2D(grid, kperm, title='kperm', plotLog=isLogNorm, allowClose=True)

        displacement = np.ceil(4.*circles[0]['r'])
        xcp = control_planes_position(grid['Lx']-displacement, displacement=displacement,
                                      control_planes=control_planes)

        permeability_data(kperm=kperm, grid=grid, circles=circles, Kfactor=Kfactor,
                          fname=filename, Kdist=Kdist, Kincl=Kincl,
                          calcKeff=False, calcKhist=True)

    if flowMethod == 'ReadVel':
        ux = np.load('./veldata/' + filename + '-ux.npy')
        uy = np.load('./veldata/' + filename + '-uy.npy')

    else:
        ux, uy = flow(grid, 1./kperm, bcc, isPeriodic=isPeriodic,
                      plotHead=plotFlow,
                      directSolver=directSolver,
                      tol=tol, maxiter=maxiter,
                      saveVel=saveVel, filename=filename)

    if calcPsi:
        psi = stream_function(grid, kperm, isPeriodic=isPeriodic,
                              plotPsi=plotPsi, saveit=savePsi)

    if dt is None:
        if integrateInTime:
            tx = 0.5*grid['Lx']/grid['Nx']/ux.max()
            ty = 0.5*grid['Ly']/grid['Ny']/uy.max()
            if Diff is not None:
                tdiff = 0.5*(grid['Lx']/grid['Nx'])*(grid['Lx']/grid['Nx'])/Diff
            else:
                tdiff = 1e9

            dt = np.min([tx, ty, tdiff,1e-3])
            print(tx) #qqq
            print(ty)
            print(tdiff)
        else:
            dt = 0.1*Kincl.min()*grid['Lx']/grid['Nx']

        print(['dt = ', dt])

    transportSolved = True

    if transportMethod == 'time':
        arrival_times, t_in_incl = transport(grid, incl_ind,
                                             Npart, ux, uy,
                                             tmax, dt, isPeriodic=isPeriodic, Diff=Diff,
                                             plotit=plotTpt, CC=kperm, InjSize=InjSize)
    elif transportMethod == 'streamlines':
        arrival_times, t_in_incl = transport_ds(grid, incl_ind,
                                                Npart, ux, uy,
                                                dt, isPeriodic=isPeriodic)
    elif transportMethod == 'pollock':
        arrival_times, t_in_incl = transport_pollock(grid, incl_ind,
                                                     Npart, ux, uy,
                                                     isPeriodic=isPeriodic,
                                                     plotit=plotTpt, CC=kperm,
                                                     xcp=xcp,
                                                     fname=filename,InjSize=InjSize)
    elif transportMethod is None:
        print("Transport not solved.\n")
        transportSolved = False
        
# BTCs and postprocess.

    if filename is None:
        filename = 'K' + str(Kfactor).replace('.', '') + pack + 'Ninc' + str(n_incl_y)


    if transportSolved:
        cbtc_time, cbtc = compute_cbtc(arrival_times,
                                       saveit=True, showfig=plotBTC, savefig=False,
                                       filename=filename)

        btc_time, btc = compute_btc(arrival_times,
                                    saveit=True, showfig=plotBTC, savefig=False,
                                    filename=filename)

        # decay reactions computed after solving transport (easier to implement).
        if reactive:
           react_rate = comp_react_rate(Rfactor, Rdist, Kincl.shape[0], Kincl, Kfactor,
                                        saveit=True, filename=filename)

           r_arrival_times = reactive_arrival_times(arrival_times,
                                                    t_in_incl, react_rate,
                                                    saveit=True, filename=filename)

           reactive_btcs(r_arrival_times, Npart,
                         saveit=True, showfig=plotBTC, savefig=False,
                         filename=filename)

        with open(filename + '.plk', 'wb') as ff:
            pickle.dump([Npart, t_in_incl, arrival_times], ff, pickle.HIGHEST_PROTOCOL)

    print("End of simulation.\n")

    if doVelPost:
        velocity_distribution(grid, kperm, ux=ux, uy=uy, incl_ind=incl_ind,
                          bins='auto', showfig=False, savefig=False,
                              savedata=True, fname=filename)
        if transportSolved:
            particle_velocity_distribution(t_in_incl, incl_ind, ux=ux, uy=uy, bins='auto',
                                           showfig=False, savefig=False,
                                           savedata=True, fname=filename)
    if doPost and transportSolved:

        postprocess(Npart, t_in_incl, arrival_times, fname=filename,
                    savedata=True, savefig=False,
                    showfig=False, figformat='pdf',
                    bins='auto', dofullpostp=False)

        print("End of postprocess.\n")

    return True
################
def setup_grid(Lx, Ny):
    """Grid set up. Given the length in x (Lx) and the
       number of cells in y (Ny) returns a numpy structured
       array  with Lx, Ly, Nx, Ny. Ly is always 1 and Nx is
       computed so that the cells are squares.
    """

    if Ny is None:
        Ny = 0

    grid = np.zeros(1, dtype={'names':['Lx', 'Ly', 'Nx', 'Ny'], \
                'formats':['float64', 'float64', 'int32', 'int32']})
    grid['Lx'] = Lx
    grid['Ly'] = 1.
    grid['Nx'] = np.int(Lx*Ny)
    grid['Ny'] = Ny

    return grid
################
def unpack_grid(grid):
    '''Retunrs the grid data in individual variables.'''
    return grid['Lx'][0], grid['Ly'][0], grid['Nx'][0], grid['Ny'][0]

################
def permeability(grid, n_incl_y, Kfactor=1., Kdist='const', pack='sqr',
                 target_incl_area=0.5, radius=None, isPeriodicK=False,
                 filename=None, plotit=False, saveit=True, overlapTol=None,
                 calcKinfo=True, calcKeff=False, calcKhist=False, control_planes=None):
    """Computes permeability parttern inside a 1. by Lx rectangle.
       The area covered by the inclusions is 1/2 of the rectanle.
       If the arrangement os random, the radius is reduced by 10%
       to increase the chance of achieving the target inclusion area.

       Then the domain is resized by adding 3*radius on the left
       to avoid boundary effects.
       The dimension of the box and the discretization are changed
       so that the grid remains regular.

       The function returns the permeability field, the cell indexes
       the location of the inclusions and the new grid.
    """

    import RecPore2D as rp

    Lx, Ly, Nx, Ny = unpack_grid(grid)

    n_incl_x = np.int(Lx*n_incl_y)
    n_incl = number_of_grains(n_incl_y, n_incl_x, pack)

    if radius is None:
        total_area = Lx*Ly
        radius = np.sqrt((target_incl_area*total_area)/(np.pi*n_incl))

    else:
        radius = np.array(radius)

    #Pore throat is computed according to regular arrangements.
    # maximum radius chosen
    if isPeriodicK:
        throat = (Ly - (2.0*n_incl_y*radius.max()))/n_incl_y
    else:
        throat = (Ly - 2.0*n_incl_y*radius.max())/(n_incl_y + 1.0)


    if pack == 'sqr' or pack == 'tri':

        pore = rp.RegPore2D(nx=n_incl_x, ny=n_incl_y,
                            radius=radius, packing=pack)
        pore.isPeriodic = isPeriodicK

        pore.bounding_box = ([0.0, 0.0, 0.5], [Lx, Ly, 1.0])
        pore.xoffset = 0.
        pore.throat = throat

    elif 'rnd' in pack:

        pore = rp.RndPore2D(lx=Lx, ly=Ly,
                            rmin=radius.min(), rmax=radius.max(),
                            target_porosity=1.-target_incl_area,
                            packing=pack)

        #maximum number of grains computed according to target porosity
        #Added 10% more grains
        ngrains_max = (target_incl_area*Lx*Ly)/(np.pi*(radius.min())**2)
        pore.ngrains_max = int(np.ceil(ngrains_max*1.1))
        print( 'Maximum number of grains set to: ' + str(pore.ngrains_max))
        pore.ntries_max = int(1e5)

        if overlapTol is None:
            pore.tolerance = 1.5*throat
        else:
            pore.tolerance = overlapTol*throat

    # Centers circles and resizes domain to avoid boundary effects
    # (2*radius added to the left and to the right).
    # Maximum radius selected in case there is a size distribution.
    ixmin = pore.circles['x'].argmin()
    ixmax = pore.circles['x'].argmax()
    xmin = pore.circles[ixmin]['x']
    rmin = pore.circles[ixmin]['r']
    rmax = pore.circles[ixmax]['r']


    #Leftmost circle's border moved to x=0.
    pore.circles[:]['x'] = pore.circles[:]['x'] - xmin + rmin
    #Space behind rightmost circle.
    xmax = pore.circles['x'].max()
    displacement = Lx - (xmax + rmin)

    #Circles centering
    pore.circles[:]['x'] = pore.circles[:]['x'] + 0.5*displacement
    #Adding of 4*r
    displacement = np.ceil(4.*rmin)
    pore.circles[:]['x'] = pore.circles[:]['x'] + 0.5*displacement

    # control planes position
    # Lx has not the displacement length.
    control_planes = control_planes_position(Lx,
                                             displacement=displacement,
                                             control_planes=control_planes)


    # if no discretization is given a 30th of the smallest inclusion's
    # radius is chosen as cell size.
    if Ny < 1:
        Ny = np.int(30/rmin)

    grid = setup_grid(Lx + displacement, Ny)

    kperm, incl_ind, Kincl = perm_matrix(grid, pore.circles, Kfactor, Kdist=Kdist)

    isLogNorm = (Kdist=='lognorm')

    if plotit:

        plot2D(grid, kperm, title='kperm', plotLog=isLogNorm, allowClose=True)

    if saveit:
        save_perm(grid, pore.circles, Kfactor, Kdist, Kincl, filename=filename)
        #pore.write_mesh(fname='g', meshtype='stl')

    if calcKinfo or calcKeff or calcKhist:
        permeability_data(grid=grid, circles=pore.circles, Kfactor=Kfactor,
                          Kdist=Kdist, Kincl=Kincl, calcKeff=calcKeff,
                          calcKhist=calcKhist, fname=filename)

    return kperm, incl_ind, grid, control_planes, Kincl

################
def flow(grid, mu, bcc, isPeriodic=True, plotHead=False,
         directSolver=True, tol=1e-10, maxiter=2000,
         saveVel=False, filename=None):
    ''' Solves the flow equation and returns the velocity
        at the cell's faces.
        mu = 1./kperm
    '''

    Lx, Ly, Nx, Ny = unpack_grid(grid)

    dx = Lx/Nx
    dy = Ly/Ny
    dx2 = dx*dx
    dy2 = dy*dy
    Np = Nx*Ny

    # Transmisibility matrices.

    Tx = np.zeros((Ny, Nx + 1))
    Tx[:, 1:Nx] = (2.*dy)/(mu[:, 0:Nx-1] + mu[:, 1:Nx])
    Ty = np.zeros([Ny + 1, Nx])
    Ty[1:Ny, :] = (2.*dx)/(mu[0:Ny-1, :] + mu[1:Ny, :])

    Tx1 = Tx[:, 0:Nx].reshape(Np, order='F')
    Tx2 = Tx[:, 1:Nx+1].reshape(Np, order='F')

    Ty1 = Ty[0:Ny, :].reshape(Np, order='F')
    Ty2 = Ty[1:Ny+1, :].reshape(Np, order='F')

    Ty11 = Ty1
    Ty22 = Ty2

    TxDirichR = (2.*dy)*(1./mu[:, Nx-1])


    if bcc == 'head':
        TxDirichL = (2.*dy)*(1./mu[:, 0])

    #Assemble system of equations
    Dp = np.zeros(Np)
    Dp = Tx1/dx2 + Ty1/dy2 + Tx2/dx2 + Ty2/dy2

    #Dirichlet b.c. on the right
    Dp[Np-Ny:Np] = Ty1[Np-Ny:Np]/dy2 + Tx1[Np-Ny:Np]/dx2 + \
                   Ty2[Np-Ny:Np]/dy2 + TxDirichR/dx2

    if bcc == 'head':
        Dp[0:Ny] = Ty1[0:Ny]/dy2 + Tx2[0:Ny]/dx2 + \
                   Ty2[0:Ny]/dy2 + TxDirichL/dx2

    #Periodic boundary conditions
    TypUp = np.zeros(Np)
    TypDw = np.zeros(Np)

    if isPeriodic:
        Typ = (2.*dx)/(mu[Ny - 1, :] + mu[0, :])

        TypDw[0:Np:Ny] = Typ
        TypUp[Ny-1:Np:Ny] = Typ

        Dp[0:Np:Ny] = Dp[0:Np:Ny] + Typ/dy2
        Dp[Ny-1:Np:Ny] = Dp[Ny-1:Np:Ny] + Typ/dy2

    # TO DO. Check if this saves memory
    import gc
    mu = None
    gc.collect()

    Am = sp.spdiags([-Tx2/dx2, -TypDw/dy2, -Ty22/dy2, Dp,
                     -Ty11/dy2, -TypUp/dy2, -Tx1/dx2],
                    [-Ny, -Ny + 1, -1, 0, 1, Ny - 1, Ny], Np, Np,
                    format='csr')

    #RHS - Boundary conditions
    u0 = 1.*dy
    hL = 1.
    hR = 0.
    S = np.zeros(Np)
    S[Np-Ny:Np+1] = TxDirichR*(hR/dx/dx) # Dirichlet X=1;
    if bcc == 'head':
        S[0:Ny] = TxDirichL*(hL/dx/dx) # Dirichlet X=0;
    else:
        S[0:Ny] = u0/dx #Neuman BC x=0;

    solved = False

    if directSolver:
        try:
            head = lgsp.spsolve(Am, S).reshape(Ny, Nx, order='F')
            solved = True

        except:
            print(grid['Ny'])
            print(grid['Nx'])
            print("spsolve out of memory.")
            solved = False



    if not solved or not directSolver:

        #head, info = lgsp.cg(Am, S,tol=1e-10)

        #black box solver
        import pyamg
        head = pyamg.solve(Am,S, maxiter=maxiter, tol=tol, verb=verbose)

        if verbose:
            print(np.linalg.norm(S-Am*head))

        head = head.reshape(Ny, Nx, order='F')

        #ml = pyamg.ruge_stuben_solver(Am)
        #head = ml.solve(S, maxiter=maxiter, tol=tol)
        #print(np.linalg.norm(S-Am*head1))
        #head = head.reshape(Ny, Nx, order='F')
        #ipdb.set_trace()

    Am = None
    S = None
    gc.collect()

    #Compute velocities

    ux = np.zeros([Ny, Nx+1])
    uy = np.zeros([Ny+1, Nx])

    if bcc == 'head':
        ux[:, 0] = -TxDirichL*(head[:, 0] - hL)/dx
    else:
        ux[:, 0] = u0

    ux[:, 1:Nx] = -Tx[:, 1:Nx]*(head[:, 1:Nx] - head[:, 0:Nx-1])/dx
    ux[:, Nx] = -TxDirichR*(hR - head[:, Nx-1])/dx

    uy[1:Ny, :] = -Ty[1:Ny, :]*(head[1:Ny, :] - head[0:Ny-1, :])/dy


    if isPeriodic:
        uy[0, :] = Typ*(head[0, :] - head[Ny - 1, :])/dy
        uy[Ny, :] = uy[0, :]


    if ux.min() < 0.:
        print('Negative ux found in ' + str(np.sum(ux<0.)) + ' cell(s)!')
        #ux[ux>0.] = 0.
    if plotHead:
        plot2D(grid, head, title='head', allowClose=True)
        plot2D(grid, ux/dy, title='ux', allowClose=True)
        plot2D(grid, uy/dx, title='uy', allowClose=True)

    if saveVel:
        fname = (filename is not None)*(filename +'-')
        np.save(fname + 'ux.npy', ux/dx)
        np.save(fname + 'uy.npy', uy/dy)

    return ux/dy, uy/dx

#####
def transport(grid, incl_ind, Npart, ux, uy, tmax, dt, Diff=None,
              isPeriodic=False, plotit=False, CC=None, InjSize=1.):
    '''Solves the transport of a line of concentration initially at the
       left boundary using a particle tracking method.

       Returns the arrival times of the particles to the right boundary
       and data about the time spent in the inclusions.'''

    Lx, Ly, Nx, Ny = unpack_grid(grid)

    if plotit and CC is None:
        CC = np.ones((Ny, Nx))

    if plotit:
        figt, axt, cbt = plot2D(grid, CC)

    t = 0.

    xp = np.zeros(Npart)
    yp = np.arange((InjSize*Ly)/Npart/2.0, InjSize*Ly, (InjSize*Ly)/Npart)
    yp = yp + 0.5 - InjSize/2.

    #qq
    #rad = 0.25231325
    #alpha = np.arange(np.pi/Npart/2.0, np.pi, np.pi/Npart)
    #alpha = np.arange(np.pi/2. + np.pi/Npart/2.0, 3.*np.pi/2., np.pi/Npart)
    #xp = 1.5 + rad*np.cos(alpha)
    #yp  = 0.5 +rad*np.sin(alpha)
    #qq
    dx = np.float(Lx/Nx)
    dy = np.float(Ly/Ny)

    Ax = ((ux[:, 1:Nx + 1] - ux[:, 0:Nx])/dx).flatten(order='F')
    Ay = ((uy[1:Ny + 1, :] - uy[0:Ny, :])/dy).flatten(order='F')

    ux = ux[:, 0:Nx].flatten(order='F')
    uy = uy[0:Ny, :].flatten(order='F')

    x1 = np.arange(0., Lx + dx, dx) #faces' coordinates
    y1 = np.arange(0., Ly + dy, dy)

    i = 0
    #ipng = 0

    lint = None

    #number of inclusions
    num_incl = incl_ind.max().astype(int)


    #time of each particle in each inclusion
    # It contains nincl dictionaries
    # Each dictionary contains the particle and the time spent.
    t_in_incl = []

    for i in range(num_incl):
        t_in_incl.append({})

    nwrite = 100#qqq # np.max([int((tmax/dt)/1000),10])
    arrival_times = np.zeros(Npart)
    i = 0

    isIn = np.where(xp < Lx)[0]
    if Diff is not None:
        Dm = np.sqrt(2.*Diff*dt)
        vv = []
        tt = []
        # random number generator
        from numpy.random import default_rng
        rng = default_rng()
        
    if tmax is None:
        print('tmax not defined. tmax = 1.0')
        tmax = 1.0
    while t <= tmax and isIn.size > 0:

        t = t + dt
        i = i + 1

        # Indexes of cells where each particle is located.
        indx = (xp[isIn]/dx).astype(int)
        indy = (yp[isIn]/dy).astype(int)

        # I thought this was faster
        #indx = np.int_(xp[isIn]/dx)
        #indy = np.int_(yp[isIn]/dy)

        ix = (indy + indx*Ny)

        t_in_incl = update_time_in_incl(t_in_incl, incl_ind, isIn,
                                        indx, indy, t*np.ones(Npart))
        # #Auxiliary array with the numbers of the inclusions
        # #where particles are. Inclusion 0 is the matrix.
        # aux1 = incl_ind[[indy], [indx]].toarray()[0]

        # #index of particles inside an inclusions
        # parts_in_incl = isIn[np.where(aux1 > 0)].astype(int)

        # #index of inclusions where particles are
        # # -1 because zero based convention of arrays...
        # incl = aux1[aux1 > 0].astype(int) - 1

        # #Update of time spent by particles in each inclusion
        # for part, inc in zip(parts_in_incl, incl):
        #     try:
        #         t_in_incl[inc][part] = t_in_incl[inc][part] + dt
        #     except:
        #         t_in_incl[inc][part] = dt

        xp[isIn] = xp[isIn] + (ux[ix] + Ax[ix]*(xp[isIn] - x1[indx]))*dt
        yp[isIn] = yp[isIn] + (uy[ix] + Ay[ix]*(yp[isIn] - y1[indy]))*dt

        if Diff is not None:
            nn = xp[isIn].shape[0]
            xp[isIn] = xp[isIn] + Dm*rng.normal(0, 1, nn) #np.random.normal(0,1,nn)
            yp[isIn] = yp[isIn] + Dm*rng.normal(0, 1, nn)

        #print ["{0:0.19f}".format(i) for i in yp]

        xp[xp < 0.] = 0.

        if isPeriodic:
            yp[yp < 0.] = yp[yp < 0.] + Ly
            yp[yp > Ly] = yp[yp > Ly] - Ly
        else:
            #boundary reflection
            yp[yp < 0.] = -yp[yp < 0.]
            yp[yp > Ly] = 2.*Ly - yp[yp > Ly]

        isOut = isIn[np.where(xp[isIn] >= Lx)]

        arrival_times[isOut] = t # Correction TO DO - (xp[isOut] - Lx)/uxp

        isIn = np.where(xp < Lx)[0]

        if i%nwrite == 0:
            print("Time: %f. Particles inside: %e" %(t, isIn.size))

            if plotit:
                #ipng = ipng + 1
                figt, axt, lint = plotXY(xp[isIn], yp[isIn], figt, axt, lint)
                axt.set_aspect('equal')
                axt.set_xlim([0., Lx])
                axt.set_ylim([0., Ly])
                plt.title(t)
                #plt.savefig(str(int(ipng)).zfill(4), dpi=600)
    print("Time: %f. Particles inside: %e" %(t, np.sum(isIn)))
    print("End of transport.")

    return arrival_times, t_in_incl

#####
def transport_ds(grid, incl_ind, Npart, ux, uy, ds, isPeriodic=False):
    '''Solves the transport of a line of concentration initially at the
       left integrating the particles' path along the streamlines..

       Returns the arrival times of the particles to the right boundary
       and data about the time spent in the inclusions.'''

    Lx, Ly, Nx, Ny = unpack_grid(grid)

    xp = np.zeros(Npart)
    yp = np.arange(Ly/Npart/2.0, Ly, Ly/Npart)

    #qq
    #rad = 0.25231325
    #alpha = np.arange(np.pi/2. + np.pi/Npart/2.0, 3.*np.pi/2., np.pi/Npart)
    #xp = 1.5 + rad*np.cos(alpha)
    #yp  = 0.5 + rad*np.sin(alpha)
    #qq
    dx = np.float(Lx/Nx)
    dy = np.float(Ly/Ny)

    Ax = ((ux[:, 1:Nx + 1] - ux[:, 0:Nx])/dx).flatten(order='F')
    Ay = ((uy[1:Ny + 1, :] - uy[0:Ny, :])/dy).flatten(order='F')

    ux = ux[:, 0:Nx].flatten(order='F')
    uy = uy[0:Ny, :].flatten(order='F')

    x1 = np.arange(0., Lx + dx, dx) #faces' coordinates
    y1 = np.arange(0., Ly + dy, dy)

    arrival_times = np.zeros(Npart)

    i = 0

    #number of inclusions
    num_incl = incl_ind.max().astype(int)


    # Time of each particle in each inclusion
    # It contains nincl dictionaries.
    # Each dictionary inclusion contain temporarily the
    # times at which the particles enters and leaves the inclusion.
    # At the end of the simulation only the time spent is stored.
    # Example:
    #
    #   t_in_incl[1] --> times for particles that visited inclusion 1.
    #   t_in_incl[1] = {1: [0.1, 3.4], 5: [1.2, 7.9]}
    #
    #   This means particle 1 entered at time 0.1 and left at time 3.4
    #   and particle 3 entered at 1.2 and left at 7.9
    #
    #   At the end of the simulation we will have
    #
    #   t_in_incl[1] = {1: 3.3, 5: 6.7}
    #
    t_in_incl = []

    for i in range(num_incl):
        t_in_incl.append({})

    isIn = np.where(xp < Lx)[0]

    while  isIn.size > 0:

        i = i + 1

        # Indexes of cells where each particle is located.
        indx = (xp[isIn]/dx).astype(int)
        indy = (yp[isIn]/dy).astype(int)

        ix = (indy + indx*Ny)

        t_in_incl = update_time_in_incl(t_in_incl, incl_ind, isIn,
                                        indx, indy, arrival_times)

        uxp = (ux[ix] + Ax[ix]*(xp[isIn] - x1[indx]))
        uyp = (uy[ix] + Ay[ix]*(yp[isIn] - y1[indy]))

        vp = np.sqrt(uxp*uxp + uyp*uyp)

        xp[isIn] = xp[isIn] + ds*uxp/vp
        yp[isIn] = yp[isIn] + ds*uyp/vp

        arrival_times[isIn] = arrival_times[isIn] + ds/vp

        out = np.where(xp[isIn] - Lx >= 0.)
        arrival_times[out] = arrival_times[out] - (xp[out] - Lx)/uxp[out]

        xp[xp < 0.] = 0.

        if isPeriodic:
            yp[yp < 0.] = yp[yp < 0.] + Ly
            yp[yp > Ly] = yp[yp > Ly] - Ly
        else:
            #boundary reflection
            yp[yp < 0.] = -yp[yp < 0.]
            yp[yp > Ly] = 2.*Ly - yp[yp > Ly]

        if i%1000 == 0:
            print("Last particle at: %e" %(np.min(xp)))

        t_in_incl = update_time_in_incl(t_in_incl, incl_ind, isIn,
                                        indx, indy, arrival_times)
        isIn = np.where(xp < Lx)[0]


    print("Particles inside: %e" %(np.sum(isIn)))
    print("End of transport.")

    return  arrival_times, t_in_incl

####
def plotXY(x, y, fig=None, ax=None, lin=None, logx=False, logy=False,
           allowClose=False):
    '''Function to do Y vs X plots. '''

    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        plt.sca(ax)

    if lin is None:
        lin, = plt.plot(x, y, 'g.')
        plt.ion()
        plt.show()
    else:
        lin.set_xdata(x)
        lin.set_ydata(y)
        fig.canvas.draw()

    if logy:
        ax.set_yscale("log", nonpositive='mask')

    if logx:
        ax.set_xscale("log", nonpositive='mask')

    if allowClose:
        input("Dale enter y cierro...")
        plt.close()
        fig = None
        ax = None
        lin = None
    return fig, ax, lin
####
def plot2D(grid, C, fig=None, ax=None, title=None, cmap='coolwarm',
           plotLog=False, allowClose=False):
    '''Function to do 2 dimensional plots of cell centerd or face centered
       varibles.'''

    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.gca()

    #Create X and Y meshgrid
    Lx, Ly, Nx, Ny = unpack_grid(grid)
    dx = Lx/Nx
    dy = Ly/Ny

    # xx, yy need to be +1 the shape of C because
    # pcolormesh needs the quadrilaterals.
    # Otherwise the last column is ignored.
    # See: matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.pcolor
    cny, cnx = C.shape

    if cnx > Nx:      # x face centered

        x1 = np.arange(-dx/2., Lx + dx, dx)
        y1 = np.arange(0., Ly + dy/2., dy)

    elif cny > Ny: # y face centered

        x1 = np.arange(0., Lx + dx/2., dx)
        y1 = np.arange(-dy/2., Ly + dy, dy)

    else:              #cell centered

        x1 = np.arange(0., Lx + dx/2., dx)
        y1 = np.arange(0., Ly + dy/2., dy)

    xx, yy = np.meshgrid(x1, y1)

    plt.sca(ax)
    plt.ion()

    if plotLog:
        C = np.log(C)

    mesh = ax.pcolormesh(xx, yy, C, cmap=cmap)

    #plt.axis('equal'
    #plt.axis('tight')
    plt.axis('scaled')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(mesh, cax=cax)

    if title is not None:
        plt.title(title)

    plt.show()

    if allowClose:
        input("Dale enter y cierro...")
        plt.close()
        fig = None
        ax = None
        cb = None

    return fig, ax, cb
################
def number_of_grains(nix, niy, pack):
    """Computes number of grains according to the packing"""

    if pack == 'sqr':
        ngrains = nix*niy
    elif pack == 'tri':
        if nix%2 == 0:
            ngrains = (nix//2)*(niy - 1) + (nix//2)*niy
        else:
            ngrains = (nix//2)*(niy - 1) + (nix//2 + 1)*(niy)
    elif pack == 'rnd':
        ngrains = nix*niy

    return ngrains
################
def load_data(filename):
    """ Loads the data in the given file.
        The file can be a text file (.dat) or
        a pickled file (.plk)
    """

    import os.path as ospath

    _, fileending = ospath.splitext(filename)

    if fileending == '.dat':
        data = np.loadtxt(filename)
        Npart = len(data)
        arrival_times = data
        return  Npart, arrival_times # Npart, t_in_incl if imported from plk

    elif fileending == '.plk':
        with open(filename, 'rb') as ff:
            data = pickle.load(ff)
            Npart = data[0]
            t_in_incl = data[1]
            arrival_times = data[2]
            return  Npart, t_in_incl, arrival_times

################
def time_per_inclusion(t_in_incl, Npart, bins='auto', saveit=False,
                       savefig=False, showfig=False, filename=None):
    """ Given the dictionary whith the time at which each particle entered
        and exited each inclusion, returns the time each particle spent
        in each inclusion in a matrix format suitable to calculate
        histograms.
        It also returns all times in a single array.
        Optionally, the data is saved as a two text files
        incl-times.dat and trap-times.dat).
    """

    #First the total time each particle spent in each inclusion is computed.
    tot_t_in_incl = total_time_in_incl(t_in_incl)

    #dictionary with the time particles spent in each inclusion.
    incl_times = {}
    trapped_part = {}
    i = 0
    for incl in tot_t_in_incl:
        vals = list(incl.values())
        if len(vals) > 0:
            incl_times[i] = np.concatenate(vals)
            trapped_part[i] = list(incl.keys())
        else:

            incl_times[i] = np.zeros(1) # so that it is an array.
        i = i + 1

    trapped_part = flatten_list(list(trapped_part.values()))
    trapped_part = np.unique(np.array(trapped_part))
    num_free_part = Npart - trapped_part.shape[0]
    trap_times = np.concatenate(np.array(list(incl_times.values()), dtype="object"))
    # adds as many zeros as free particles
    trap_times = np.concatenate((trap_times, np.zeros(num_free_part)))

    if saveit:

        if filename is None:
            fname = 'incl-times.dat'
        else:
            fname = filename + '-incl-times.dat'

        from itertools import zip_longest

        import csv
        with open(fname, 'w') as ff:
            writer = csv.writer(ff, delimiter=' ')
            # I whish I knew why this works...
            for values in zip_longest(*list(incl_times.values())):
                writer.writerow(np.asarray(values))

        if filename is None:
            fname = 'trap-times.dat'
        else:
            fname = filename + '-trap-times.dat'

        np.savetxt(fname, trap_times)

        if filename is None:
            figname = 'trap-dist'
        else:
            figname = filename + '-trap-dist'
        plot_hist(trap_times, title='', bins=bins,
                  showfig=showfig, savefig=savefig,
                  savedata=saveit, figname=figname)

        figname = figname + '-no-zero'
        plot_hist(trap_times[trap_times>0.], title='', bins=bins,
                 showfig=showfig, savefig=savefig,
                 savedata=saveit, figname=figname)

    return incl_times, trap_times

################
def inclusions_histograms(incl_times, showfig=True, savefig=False,
                          savedata=False, fname='', figformat='pdf',
                          bins='auto'):
    """Plots and saves the histogram for all inclusions."""

    n_incl = len(incl_times)
    i = 0
    for incl in incl_times:

        times = incl_times[incl].flatten()

        title = 'inclusion ' + str(i + 1) + '/' + str(n_incl)
        figname = fname + '-incl-hist-' + str(i) + '.pdf'
        plot_hist(times, title=title, bins=bins,
                  showfig=showfig, savefig=savefig, savedata=savedata,
                  figname=figname, figformat=figformat)
        i = i + 1
    return True

################
def plot_hist(data, title='', bins='auto', density=True, showfig=True,
              savefig=False, savedata=False, figname='zz', figformat='pdf'):
    '''Plots the histogram of the data.'''

    vals, edges = np.histogram(data, bins=bins, density=density)
    if showfig:
        plotXY((edges[:-1] + edges[1:])/2., vals, allowClose=not savefig)

        if savefig:
            save_fig(xlabel='time', ylabel='freq', title='title',
                     figname=figname, figformat=figformat)

    if savedata:
        filename = figname + '.dat'
        np.savetxt(filename, np.matrix([edges[:-1], edges[1:], vals]).transpose())

    return True
################
def postprocess_from_file(fname, savedata=True, savefig=False,
                          showfig=False, figformat='pdf',
                          bins='auto', dofullpostp=False):
    """Post process from plk file. Computes the histograms for
       particles and inclusions. Saves data and/or figures.
    """

    Npart, t_in_incl, arrival_times = load_data(fname + '.plk')

    postprocess(Npart, t_in_incl, arrival_times, fname=fname,
                savedata=savedata, savefig=savefig,
                showfig=showfig, figformat=figformat,
                bins=bins, dofullpostp=dofullpostp)
    return True

################
def postprocess(Npart, t_in_incl, arrival_times, fname='',
                savedata=True, savefig=False,
                showfig=False, figformat='pdf',
                bins='auto', dofullpostp=False):
    """Post process simulation data.
    """
    compute_cbtc(arrival_times, bins='auto', saveit=savedata,
                 showfig=showfig, savefig=savefig,
                 filename=fname)

    compute_btc(arrival_times, bins='auto', saveit=savedata,
                showfig=showfig, savefig=savefig,
                filename=fname)

    t_mobile, t_immobile = mobile_immobile_time(t_in_incl, arrival_times,
                                                filename=fname,
                                                saveit=savedata)
    figname = fname + '-immob-hist'
    plot_hist(t_immobile, title='', bins='auto',
              showfig=showfig, savefig=savefig,
              savedata=savedata, figname=figname)

    figname = fname + '-mob-hist'
    plot_hist(t_mobile, title='', bins='auto',
              showfig=showfig, savefig=savefig,
              savedata=savedata, figname=figname)

    incl_times, trap_times = time_per_inclusion(t_in_incl, Npart,
                                                saveit=savedata,
                                                filename=fname)

   # TO DO: Verify.
   # _, _ = incl_per_time(t_in_incl, plotit=showfig,
   #                          saveit=savedata, filename=fname)

    free_trapped_arrival(arrival_times, t_immobile, saveit=savedata,
                         filename=fname)

    inclusion_per_particle(t_in_incl, Npart, saveit=savedata, filename=fname)

    if dofullpostp:
    #particle histogram
        plot_hist(trap_times, title=fname + ' particles', bins=bins,
                  showfig=showfig, savefig=savefig, savedata=savedata,
                  figname=fname + '-part-hist', figformat=figformat)


        inclusions_histograms(incl_times, showfig=showfig, savefig=savefig,
                              savedata=False, fname=fname,
                              figformat=figformat, bins=bins)

    return True
################
def postprocess_all(savedata=True, savefig=False,
                    showfig=False, figformat='pdf',
                    bins='auto', dofullpostp=False):

    """ Post process all the cases in a folder."""

    import os as os

    files = os.listdir()
    for file in files:
        if file.endswith('plk'):
            fname = os.path.splitext(file)[0]
            print(fname + '...')
            postprocess_from_file(fname, savedata=savedata, savefig=savefig,
                                  showfig=showfig, figformat='pdf',
                                  bins='auto', dofullpostp=dofullpostp)
    return True
################
def stream_function(grid, kperm, isPeriodic=False, plotPsi=False,
                    saveit=False, filename=None):
    '''Compute the stream function.
    isPeriodic = False:
      The stream function is prescribed at the boundaries so that
      it is equivalent to prescribed flow on the left and right
      boundaries and no flow on top and bottom boundaries.
    isPeriodic = True:
     TO DO'''
    if isPeriodic==True:
        print('')
        print('Periodic boundary conditions in stream function not implemented.')
        print('No flow conditions used.')
        print('')

    Lx, Ly, Nx, Ny = unpack_grid(grid)

    dx = Lx/(Nx-1)
    dy = Ly/(Ny-1)
    Np = Nx*Ny

    D1x, D2x, D1y, D2y = fd_mats(Nx, Ny, dx, dy, isPeriodic=isPeriodic)

    Y = np.log(kperm).reshape(Np, order='F')
    #Kmat = (D1x.multiply((D1x*Y).reshape(Np, 1)) +
    #        D1y.multiply((D1y*Y).reshape(Np, 1)))

    Amat = (D2y + D2x - (D1x.multiply((D1x*Y).reshape(Np, 1)) +
                         D1y.multiply((D1y*Y).reshape(Np, 1)))).tolil()

    RHS = np.zeros(Np)
    #BC
    #
    #Top boundary
    Amat[0:Np:Ny, :] = 0.0
    idx = np.arange(0, Np, Ny)
    Amat[idx, idx] = 1.0
    RHS[0:Np:Ny] = Ly
    #
    #Bottom boundary
    Amat[Ny-1:Np:Ny, :] = 0.0
    idx = np.arange(Ny-1, Np, Ny)
    Amat[idx, idx] = 1.0
    RHS[Ny-1:Np:Ny] = 0.0
    #
    if isPeriodic:
        RHS[0:Np:Ny] = RHS[0:Np:Ny] - Ly/2.0
        RHS[Ny-1:Np:Ny] = RHS[Ny-1:Np:Ny] - Ly/2.0

    #Left boundary
    Amat[0:Ny, :] = 0.0
    idx = np.arange(0, Ny, 1)
    Amat[idx, idx] = 1.0
    RHS[0:Ny] = np.linspace(0., Ly, Ny)[::-1]
    if isPeriodic:
        RHS[0:Ny] = np.linspace(0., Ly, Ny)[::-1] - Ly/2.0
    #
    #Right boundary
    Amat[Np-Ny:Np, :] = 0.0
    idx = np.arange(Np-Ny, Np, 1)
    Amat[idx, idx] = 1.0
    RHS[Np-Ny:Np] = RHS[0:Ny]

    psi = lgsp.spsolve(Amat.tocsr(), RHS).reshape(Ny, Nx, order='F')
    if plotPsi:
        plot_stream(psi, grid, kperm=kperm, circles=None)

    if saveit:
        fname = 'psi.npy'
        if filename is not None:
            fname = filename + '-' + fname
        np.save(fname, psi)

    return psi
################
def plot_stream(psi, grid, kperm=None, circles=None, N=50, cmap='coolwarm'):
    ''' plot stream function on top of permeability.
    '''

    fig = plt.figure()
    ax = fig.gca()
    if kperm is not None:
        fig, ax, cb = plot2D(grid, kperm, fig=fig, ax=ax, cmap=cmap)
        cb.remove()

    if circles is not None:
        for c in circles:
            circle1 = plt.Circle((c['x'], c['y']), c['r'], color='k', fill=False)
            ax.add_artist(circle1)

    Lx, Ly, Nx, Ny = unpack_grid(grid)
    dx = Lx/(Nx-1)
    dy = Ly/(Ny-1)

    x1 = np.arange(0.0, Lx+dx/2., dx)
    y1 = np.arange(0.0, Ly+dy/2., dy)
    xx, yy = np.meshgrid(x1, y1)
    ax.contour(xx, yy, np.abs(psi), N, linewidths=1.0, colors='k')
    plt.axis('scaled')
    plt.show()
    input("Dale enter y cierro...")
    plt.close()

    return True
################
def fd_mats(Nx, Ny, dx, dy, isPeriodic=False):
    '''Computes finite differeces matrices.'''
    Np = Nx*Ny
    dx2 = dx*dx
    dy2 = dy*dy

    # First derivatives in Y
    Dp = np.zeros(Np)
    Dup = np.ones(Np)/(2.0*dy)
    Ddw = -np.ones(Np)/(2.0*dy)
    DperUp = np.zeros(Np)
    DperDw = np.zeros(Np)
    #
    #Top boundary
    Ddw[Ny-1:Np:Ny] = 0.0 #diagonal inferior
    if isPeriodic:
        DperUp[Ny-1:Np:Ny] = -1.0/(2.0*dy)
    else:
        Dp[0:Np:Ny] = -1.0/dy #diagonal
        Dup[1:Np:Ny] = 1.0/dy #diagonal superior
    #
    #Bottom bounday
    Dup[Ny:Np:Ny] = 0.0
    if isPeriodic:
        DperDw[0:Np:Ny] = 1.0/(2.0*dy)
    else:
        Dp[Ny-1:Np:Ny] = 1.0/dy #diagonal
        Ddw[Ny-2:Np:Ny] = -1.0/dy #diagonal inferior
    #
    D1y = sp.spdiags([DperDw, Ddw, Dp, Dup, DperUp],
                     [-Ny+1, -1, 0, 1, Ny-1], Np, Np,
                     format='csr')
    #
    #
    # First derivatives in X
    Dp = np.zeros(Np)
    Dup = np.ones(Np)/(2.0*dx)
    Ddw = -np.ones(Np)/(2.0*dx)
    #
    #Left boundary
    Dp[0:Ny] = -1.0/dx #diagonal
    Dup[Ny:2*Ny] = 1.0/dx #diagonal superior
    #
    # Right boundary
    Dp[Np-Ny:Np] = 1.0/dx #diagonal
    Ddw[Np-2*Ny:Np] = -1.0/dx #diagonal inferior
    #
    D1x = sp.spdiags([Ddw, Dp, Dup], [-Ny, 0, Ny], Np, Np, format='csr')

    #Second derivative in Y
    Dp = -2.0*np.ones(Np)/dy2
    Ddw = np.ones(Np)/dy2
    Dup = np.ones(Np)/dy2
    Dup2 = np.zeros(Np) #segunda diagonal superior
    Ddw2 = np.zeros(Np) #segunda diagonal inferior
    DperUp = np.zeros(Np)
    DperDw = np.zeros(Np)
    #
    # Top boundary
    if isPeriodic:
        DperUp[Ny-1:Np:Ny] = 1.0/(dy2)
        Ddw[Ny-1:Np:Ny] = 0. # diagonal inferior
    else:
        Dp[0:Np:Ny] = 1.0/dy2 #diagonal
        Ddw[Ny-1:Np:Ny] = 0. # diagonal inferior
        Dup[1:Np:Ny] = -2.0/dy2 #diagonal superior
        Dup2[2:Np:Ny] = 1.0/dy2
    #
    # Bottom boundary
    if isPeriodic:
        DperDw[0:Np:Ny] = 1.0/(dy2)
        Dup[Ny:Np:Ny] = 0.0 #Diagonal superior
    else:
        Dp[Ny-1:Np:Ny] = 1.0/dy2 #diagonal
        Ddw[Ny-2:Np:Ny] = -2.0/dy2 #diagonal inferior
        Ddw2[Ny-3:Np:Ny] = 1.0/dy2
        Dup[Ny:Np:Ny] = 0.0 #Diagonal superior

    D2y = sp.spdiags([Ddw2, Ddw, Dp, Dup, Dup2],
                     [-2, -1, 0, 1, 2],
                     Np, Np, format='csr')
    if isPeriodic:
        D2y = D2y + sp.spdiags([DperDw, DperUp],
                               [-Ny + 1, Ny - 1],
                               Np, Np, format='csr')

    #Second derivative in X
    Dp = -2.0*np.ones(Np)/dx2
    Ddw = np.ones(Np)/dx2
    Dup = np.ones(Np)/dx2
    #
    # Left boundary
    Dp[0:Ny] = 1.0/dx2 #diagonal
    Dup[Ny:2*Ny] = -2.0/dx2 #diagonal superior
    Dup2 = np.zeros(Np) #segunda diagonal superior
    Dup2[2*Ny:3*Ny] = 1.0/dx2
    #
    # Right boundary
    Dp[Np-Ny:Np] = 1.0/dx2 #diagonal
    Ddw[Np-2*Ny:Np] = -2.0/dx2 #diagonal inferior
    Ddw2 = np.zeros(Np) #segunda diagonal inferior
    Ddw2[Np-3*Ny:Np-2*Ny] = 1.0/dx2

    D2x = sp.spdiags([Ddw2, Ddw, Dp, Dup, Dup2],
                     [-2*Ny, -Ny, 0, Ny, 2*Ny],
                     Np, Np, format='csr')

    return D1x, D2x, D1y, D2y
################
def update_time_in_incl(t_in_incl, incl_ind, isIn, indx, indy, time):
    """ Given the particles inside the domain (isIn), the indexes of
        the cells where the particles are (indx, indy), the indexes
        of the cells inside inclusions (incl_ind), and the current time
        (per particle if integrating along streamlines) (time), updates
        the time the particle entered and/or exited each inclusion.
    """
    #Auxiliary array with the numbers of the inclusions
    #where particles are. Inclusion 0 is the matrix.
    aux1 = incl_ind[[indy], [indx]].toarray()[0]

    #index of particles inside an inclusion
    parts_in_incl = isIn[np.where(aux1 > 0)].astype(int)

    #index of inclusions where particles are
    # -1 because zero based convention of arrays...
    incl = aux1[aux1 > 0].astype(int) - 1
    #ESTO DA PROBLEMAS SI LA PARTÍCULA ESTÁ MENOS DE UN DT EN LA INCLUSIÓN (TRANSPORTE EULERIANO)
    # O SI SÓLO ATRAVIESA UNA CELDA DENTRO DE LA INCLUSIÓN (POLLOCK).
    #Update of time spent by particles in each inclusion
    for part, inc in zip(parts_in_incl, incl):
        try:
            t_in_incl[inc][part][1] = time[part]
        except:
            # We use the error to initialize t_in_incl[inc][part] the first time.
            #ipdb.set_trace()
            t_in_incl[inc][part] = [time[part], 0.0]

    return t_in_incl
################
def total_time_in_incl(t_in_incl):
    """ Given the dictionary whith the time at which each particle entered
        and exited each inclusion, returns the total time spent for each
        particle in each inclusion.
    """
    # total time of particles in inclusions
    num_incl = len(t_in_incl)
    incl = np.arange(0, num_incl, 1)
    tot_t_in_incl = t_in_incl.copy()

    for dic1, incl in zip(tot_t_in_incl, incl):
        tot_t_in_incl[incl] = {k:np.diff(v) for k, v in dic1.items()}

    return tot_t_in_incl
################
def compute_cbtc(arrival_times, Npart=None, bins=None, saveit=False,
                 logx=False, logy=False, showfig=False,
                 savefig=False, filename=None):
    ''' Cummulative breakthrough curve from arrival times.'''
    if Npart is None:
        Npart = arrival_times.shape[0]

    nVals = arrival_times.shape[0]

    if bins is None:
        cbtc_time = np.sort(arrival_times)
        cbtc = nVals/Npart - np.cumsum(np.ones(nVals))/Npart
    else:
        vals, edges = np.histogram(arrival_times, bins=bins, density=False)
        cbtc_time = (edges[:-1] + edges[1:])/2.
        cbtc = nVals/Npart - np.cumsum(vals)/Npart

    if saveit:
        fname = 'cbtc' + (bins is not None)*'-h' + '.dat'
        if filename is not None:
            fname = filename + '-' + fname

        np.savetxt(fname, np.matrix([cbtc_time, cbtc]).transpose())
    if showfig:
        plotXY(cbtc_time, cbtc, logx=logx, logy=logy, allowClose=True)

        if savefig:
            save_fig(xlabel='time', ylabel='cbtc', title='',
                     figname=fname, fogformat='pdf')

    return cbtc_time, cbtc

################
def mobile_immobile_time(t_in_incl, arrival_times, filename=None, saveit=False):

    #adds up the time spent in each inclusion

    t_immobile = np.zeros(arrival_times.shape[0])
    for incl in t_in_incl:
        for part in incl:
            t_immobile[part] = t_immobile[part] + np.diff(incl[part])

    t_mobile = arrival_times - t_immobile

    if saveit:
        fname = 'mob.dat'
        if filename is not None:
            fname = filename + '-' + fname

        np.savetxt(fname, t_mobile)

        fname = 'immob.dat'
        if filename is not None:
            fname = filename + '-' + fname

        np.savetxt(fname, t_immobile)

    return t_mobile, t_immobile
################
def perm_matrix(grid, circles, Kfactor, Kdist='const', Kincl=None):
    """Creates the permeability matrix and indexes of inclusions
        given a set of circles."""

    Lx, Ly, Nx, Ny = unpack_grid(grid)

    kperm = np.ones([Ny, Nx])
    incl_ind = np.zeros([Ny, Nx])
    dx = Lx/Nx
    dy = Ly/Ny
    x1 = np.arange(dx/2.0, Lx, dx) #cell centers' coordinates
    y1 = np.arange(dy/2.0, Ly, dy)
    xx, yy = np.meshgrid(x1, y1)

    n_incl = circles.shape[0]

    if  Kincl is None:
        Kincl = calc_distributed_perm(Kfactor, Kdist, n_incl)
    else:
        print('Using given Kincl.')
    i = 0

    for circ in circles:
        x0 = circ['x']
        y0 = circ['y']
        r = circ['r']
        mask = ((xx - x0)**2.0 + (yy - y0)**2.0) < r**2.0
        kperm[mask] = Kincl[i]
        incl_ind[mask] = i + 1 #0 is the index for the matrix
        i = i + 1

    return kperm, sp.csr_matrix(incl_ind), Kincl
################
def load_perm(fname):
    """ Loads the permeability distribution in the given plk file."""

    with open(fname + '-perm.plk', 'rb') as ff:
        data = pickle.load(ff)
        grid = data[0]
        circles = data[1]
        Kfactor = np.array(data[2])
        # For compatibility with previus versions.
        try:
            Kdist = data[3]
            Kincl = data[4]

        except:
            Kdist = 'const'
            Kfactor = np.atleast_1d(Kfactor)
            Kincl = np.full(circles.shape[0], Kfactor[0])

    return grid, circles, Kfactor, Kdist, Kincl
################
def load_react(fname):
    """ Loads the reaction rate distribution in the given plk file."""

    with open(fname + '-react.plk', 'rb') as ff:
        data = pickle.load(ff)
        Rfactor = data[0]
        Rdist = data[1]
        react_rate = np.array(data[2])

    return Rfactor, Rdist, react_rate

################
def incl_per_time(t_in_incl, plotit=False, saveit=False, filename=None):
    """ Given the dictionary whith the time at which each particle entered
        and exited each inclusion, returns the total number of inclusions
        that contain at least one particle at a given time.
    """

    # total time of particles in inclusions
    #num_incl = len(t_in_incl)
    #incl_indx = np.arange(0, num_incl, 1)

    # We build an array with all residence times
    # (entrance and exit of each particle in each visited inclusion).
    residence_times = [list(d.values()) for d in t_in_incl]

    residence_times = np.asarray(flatten_list(residence_times))

    #Latest timeat which an inclusion was occupied.
    tmax = residence_times.max()

    # Then we divide the time interval in 1000 parts.
    times = np.arange(0.0, tmax + tmax/10, tmax/10)
    occ_incl = np.zeros(times.shape[0])
    i = 0

    #For each time we check how many inclusions are occupied.
    for t in times:
        a1 = np.array(residence_times[:, 0] < t)
        a2 = np.array(residence_times[:, 1] > t)
        occ_incl[i] = np.sum(a1*a2)
        #isOccupied = np.where( (residence_times[:, 0] < t)
        #                     & (residence_times[:, 1] > t))
        #occ_incl[i] = np.sum(np.asarray(isOccupied)>0)
        i = i + 1

    if plotit:
        plotXY(times, occ_incl, allowClose=True)

    if saveit:
        fname = 'occ-incl.dat'

        if filename is not None:
            fname = filename + '-' + fname

        np.savetxt(fname, np.matrix([times, occ_incl]).transpose())

    return times, occ_incl
################
def plot_perm_from_file(fname, plotWithCircles=True, faceColor='g',
                        edgeColor='k', fill=False, axisColor='k',
                        backgroundColor='w', showTicks=True,
                        allowClose=True, showFig=True, saveFig=False,
                        removeBuffer=False,isLog=False, cmap='coolwarm'):
    '''Load permeability data from plk file and plots it.
       The color options allow to generate a image that can be used to
       compute the inclusion distribution properties. For example,
       plot_perm_from_file('3rnd', faceColor='w', edgeColor='w',
                           backgroundColor='k', showTicks=False)

      shows the inclusions as white circles with black background.
    '''
    #TO DO : check that file exists.
    grid, circles, Kfactor, Kdist, Kincl = load_perm(fname)

    if removeBuffer:
        radius = circles[0]['r']
        displacement = np.ceil(4.*radius)
        grid = setup_grid(grid['Lx'] - displacement, grid['Ny'])
        circles[:]['x'] = circles[:]['x'] - 0.5*displacement

    if plotWithCircles:
        fig = plt.figure()
        ax = fig.gca()
        i = 0

        if fill:
            if isLog:
                Kincl = np.log(Kincl)

            import matplotlib.cm as cm
            from matplotlib.colors import Normalize
            norm = Normalize(Kincl.min(), Kincl.max())
            #cmap = cm.jet
            cmap_fill = cm.ScalarMappable(norm=norm,  cmap=plt.get_cmap(cmap))
            cmap_fill.set_array([])

        color = None

        for c in circles:

            if fill:
                color = cmap_fill.cmap(norm(Kincl[i]))
                i = i +1

            circle1 = plt.Circle((c['x'], c['y']), c['r'],
                                 edgecolor=edgeColor,
                                 facecolor=faceColor,
                                 color=color,
                                 fill=fill)

            ax.add_artist(circle1)

        plt.axis('scaled')
        plt.xlim(0, grid['Lx'])
        plt.ylim(0, grid['Ly'])

        ax.set_facecolor(backgroundColor)

        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(0.5)
            ax.spines[axis].set_color(axisColor)
            ax.spines[axis].set_visible(True)

        plt.ion()
        if not showTicks:
            ax.set_xticks([])
            ax.set_yticks([])

        if fill:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(cmap_fill, cax=cax)

        if showFig:
            plt.show()

        if saveFig:
            import os.path as ospath
            figname = ospath.split(fname)[1] + '-perm.pdf'
            plt.savefig(figname, format='pdf')


        if allowClose:
            input("Dale enter y cierro...")

        plt.close()

    else:
        kperm, _, _ = perm_matrix(grid, circles, Kfactor, Kdist=Kdist, Kincl=Kincl)

        plot2D(grid, kperm, title='K', plotLog=isLog, allowClose=True)

################
def free_trapped_arrival(arrival_times, t_immobile, bins='auto',
                         saveit=False, logx=False, logy=False,
                         showfig=False, savefig=False, filename=None):
    """Given the arrival time and the immobile time of all particles,
        returns the arrival times of particles that visited at least
        one inclusion (trapped) and of those that did not visit any
        inclusion (free).
        Optionally, the data is saved as a text file.

    """
    if filename is None:
        filename = ''
    else:
        filename = filename + '-'

    trapped = t_immobile > 0.

    arrivals_trapped = arrival_times[trapped]
    arrivals_free = arrival_times[~ trapped]

    traptime, trapcbtc = compute_cbtc(arrivals_trapped, bins=bins,
                                      saveit=saveit, logx=logx, logy=logy,
                                      showfig=showfig, savefig=savefig,
                                      filename=filename + 'trap')

    freetime, freecbtc = compute_cbtc(arrivals_free, bins=bins,
                                      saveit=saveit, logx=logx, logy=logy,
                                      showfig=showfig, savefig=savefig,
                                      filename=filename + 'free')


    return traptime, trapcbtc, freetime, freecbtc
################
def inclusion_per_particle(t_in_incl, Npart, saveit=False, showfig=False,
                           savefig=False, filename=None):
    """ Given the dictionary whith the time at which each particle entered
        and exited each inclusion, returns the number of inclusions visited
        by each particle.
        Optionally, the data is saved as a text file.
    """
    incl_per_part = np.zeros(Npart)
    for incl in t_in_incl:
        incl_per_part[list(incl.keys())] = incl_per_part[list(incl.keys())] + 1

    if saveit:
        fname = 'visited-incl.dat'
        if filename is not None:
            fname = filename + '-' + fname

        np.savetxt(fname, incl_per_part)

        fname = 'trap-events'
        if filename is not None:
            fname = filename + '-' + fname

    if saveit or showfig:

        num_incl = len(t_in_incl)
        bins = np.arange(0, num_incl + 2)
        plot_hist(incl_per_part, title='', bins=bins, density=True,
                  showfig=showfig, savefig=savefig, savedata=saveit,
                  figname=fname)

        plot_hist(incl_per_part[incl_per_part>0], title='', bins=bins,
                  showfig=showfig, savefig=savefig, savedata=saveit,
                  figname=fname + '-no-zero')

    print("Trapping Events.")
    print("Mean: ", np.mean(incl_per_part), "Variance: ", np.var(incl_per_part))


    return incl_per_part
################
def save_fig(xlabel='', ylabel='', title='', figname='', figformat='pdf'):
    '''Saves current figure.'''

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if figformat == 'pdf':
        plt.savefig(figname + '.pdf', format='pdf')
    if figformat == 'tikz':
        from matplotlib2tikz import save as tikz_save
        tikz_save(figname + '.tex')

    input("Dale enter y cierro...")
    plt.close()
################
def equivalent_permeability(grid, kperm, isPeriodic=False,
                            directSolver=False, tol=1e-10, maxiter=2000):
    """Compute the  equivalent permeability of the medium.
    """
    bcc = 'head'
    ux, _ = flow(grid, 1./kperm, bcc, isPeriodic=isPeriodic, plotHead=False,
                 directSolver=directSolver, tol=tol, maxiter=maxiter)

    dy = grid['Ly']/grid['Ny']
    gradH = 1./grid['Lx']
    Q = np.sum(ux[:, -1])*dy

    return Q/(grid['Ly']*gradH)
################
def equivalent_permeability_from_file(fname):
    """Compute the equivalent permeability of the medium.
       Data read from perm.plk file.
    """
    grid, circles, Kfactor, Kdist, Kincl = load_perm(fname)
    kperm, _, _ = perm_matrix(grid, circles, Kfactor, Kdist=Kdist, Kincl=Kincl)
    print('')
    print('equivalent_permeability_from_file not implemented')

    return equivalent_permeability(grid, kperm)
################
def inclusion_area(grid, circles,  kperm=None):
    '''Computes inclusions area.'''

    if kperm is None:
        kperm, _, _ = perm_matrix(grid, circles, 0.1)

    dx = grid['Lx']/grid['Nx']
    # We do not take into account the displacement
    radius = circles[0]['r']
    displacement = np.ceil(4.*radius)
    ndisp = np.int(displacement/dx/2)

    incl_area = sum(kperm[:, ndisp:-ndisp].flatten() < 1.)
    total_area = sum(kperm[:, ndisp:-ndisp].flatten() > 0.)

    return incl_area/total_area
################
def flatten_list(l):
    '''Merge items in list with sublists.
       From
       https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
    '''
    return  [item for sublist in l for item in sublist]
################
def average_number_of_inclusions(kperm):
    ''' Computes the average number of inclusions in the vertical and
        horizontal directions'''

    h_aux = np.sum(np.abs(np.diff(kperm, axis=1)) > 0, axis=1)/2
    v_aux = np.sum(np.abs(np.diff(kperm, axis=0)) > 0, axis=0)/2

    #mean in the whole domain
    h_avg = np.mean(h_aux)
    v_avg = np.mean(v_aux)

    #mean removing zeros
    h_avg_nonzero = np.nanmean(np.where(h_aux != 0, h_aux, np.nan))
    v_avg_nonzero = np.nanmean(np.where(v_aux != 0, v_aux, np.nan))

    return h_avg, v_avg, h_avg_nonzero, v_avg_nonzero
################
def permeability_data_from_file(fname=None, calcKeff=False, calcKhist=False,
                      directSolver=False, tol=1e-10, maxiter=2000):
    ''' Loads permeability and  computes statistics.
    '''

    grid, circles, Kfactor, Kdist, Kincl = load_perm(fname)
    kperm, _, Kincl = perm_matrix(grid, circles, Kfactor, Kdist=Kdist, Kincl=Kincl)
    permeability_data(grid=grid, circles=circles, Kfactor=Kfactor, fname=fname,
                      Kdist=Kdist, Kincl=Kincl, calcKeff=calcKeff, calcKhist=calcKhist,
                      directSolver=directSolver, tol=tol, maxiter=maxiter)

    return
################
def permeability_data(kperm=None, grid=None, circles=None, Kfactor=None, fname=None,
                      Kdist='const', Kincl=None, calcKeff=False, calcKhist=False,
                      directSolver=False, tol=1e-10, maxiter=2000):

    print('Permeability data.')
    print('============ ====')

    if kperm is None:
        print('I need to compute the permeability matrix.')

        if grid is not None and circles is not None and Kfactor is not None and Kdist is not None:
            kperm, _, Kincl = perm_matrix(grid, circles, Kfactor, Kdist=Kdist, Kincl=Kincl)
        else:
            print('Not enough data to compute the permeability matrix.')

    if kperm is not None:

        print('Lx = '+ str(grid['Lx']))
        print('Ny = '+ str(grid['Ny']))
        print('radius = '+ str(circles[0]['r']))
        print('inclusions = '+ str(circles.shape[0]))
        incl_area = inclusion_area(grid, circles, kperm=kperm)
        print('inclusion_area = ' + str(incl_area))

        h_avg, v_avg, h_avg_nonzero, v_avg_nonzero = average_number_of_inclusions(kperm)
        print('Average number of inclusions:')
        print('Horizontal = ' + str(h_avg) + ' (' + str(h_avg_nonzero) + ')')
        print('Vertical = ' + str(v_avg) + ' (' + str(v_avg_nonzero) + ')')

        print('Kfactor = '+ str(Kfactor))
        print('Kdist = ' + Kdist)
        print('K mean and variance: ' + str(Kincl.mean()) + ' ' + str(Kincl.var()))
        print('logK mean and variance: ' + str(np.log(Kincl).mean()) + ' ' + str(np.log(Kincl).var()))
        if calcKhist:
            figname = fname + '-perm-h'
            plot_hist(Kincl, title='', bins='auto', density=True,
                      showfig=False, savefig=False, savedata=True,
                      figname=figname)
            print('Inclusions permeability histogram computed.')

        if calcKeff:
            keff = equivalent_permeability(grid, kperm,
                                           directSolver=directSolver,
                                           tol=tol, maxiter=maxiter)
            print('Keff = ' + str(keff))
        else:
            keff = None
            print('Equivalent permeability not computed.')

        return grid['Lx'], circles[0]['r'], circles.shape[0], incl_area, keff
################
def get_mesh(grid, centering='cell'):
    """ Returns face centered, x centered or y centered mesh.
        centering = cell, x, y.
    """

    Lx, Ly, Nx, Ny = unpack_grid(grid)
    dx = Lx/Nx
    dy = Ly/Ny

    if centering == 'x':     # x face centered

        x = np.arange(0., Lx + dx, dx)
        y = np.arange(0., Ly + dy/2., dy)

    elif centering == 'y':      # y face centered

        x = np.arange(0., Lx + dx/2., dx)
        y = np.arange(0., Ly + dy, dy)

    elif centering == 'cell':  #cell centered

        x = np.arange(dx, Lx + dx/2., dx)
        y = np.arange(dy, Ly + dy/2., dy)

    xx, yy = np.meshgrid(x, y)

    return xx, yy, x, y
################
def velocity_distribution(grid, kperm, ux=None, uy=None, incl_ind=None,
                          bins='auto', showfig=False, savefig=False,
                          savedata=False, fname='', directSolver=True,
                          tol=1e-10, maxiter=2000):

    """Computes the velocity distribution in the inclusions.
    """

    xx, _, _, _ = get_mesh(grid)
    xmax = np.max(xx[kperm < 1.])
    xmin = np.min(xx[kperm < 1.])

    if ux is None or uy is None:
        ux, uy = flow(grid, 1./kperm, 'flow', isPeriodic=True, plotHead=False,
                      directSolver=directSolver, tol=tol, maxiter=maxiter)

    uxm = (ux[:, 0:-1] + ux[:, 1:])/2.
    uym = (uy[0:-1, :] + uy[1:, :])/2.

    print('max ux: ' +  str(ux.max()))
    print('min ux: ' +  str(ux.min()))
    print('max uy: ' +  str(uy.max()))
    print('min uy: ' +  str(uy.min()))

    vel = np.sqrt(uxm*uxm + uym*uym)

    incl_ind = incl_ind.todense().astype(int)

    figname = fname + '-vel-incl'
    #kperm < 1.0
    plot_hist(vel[incl_ind > 0], title='', bins=bins, density=True,
              showfig=showfig, savefig=savefig, savedata=savedata,
              figname=figname, figformat='pdf')

    figname = fname + '-vel-mat'

    plot_hist(vel[incl_ind == 0], title='', bins=bins, density=True,
              showfig=showfig, savefig=savefig, savedata=savedata,
              figname=figname, figformat='pdf')


    print('Velocity in matrix')
    print('mean ux: ' +  str(uxm[incl_ind == 0].mean()))
    print('mean uy: ' +  str(uym[incl_ind == 0].mean()))
    print('mean v: ' +  str(vel[incl_ind == 0].mean()))

    figname = fname + '-vel-mat-no-buffer'
    #mask = (kperm > 0.99) & (xx > xmin) & (xx < xmax)
    mask = (incl_ind == 0) & (xx > xmin) & (xx < xmax)

    plot_hist(vel[mask], title='', bins=bins, density=True,
              showfig=showfig, savefig=savefig, savedata=savedata,
              figname=figname, figformat='pdf')

    print('Velocity in matrix (no buffer)')
    print('mean ux: ' +  str(uxm[mask].mean()))
    print('mean uy: ' +  str(uym[mask].mean()))
    print('mean v: ' +  str(vel[mask].mean()))

    figname = fname + '-vel-all'

    plot_hist(vel, title='', bins=bins, density=True,
              showfig=showfig, savefig=savefig, savedata=savedata,
              figname=figname, figformat='pdf')

    # Statistics by inclusion
    if incl_ind is not None:
        num_incl = incl_ind.max().astype(int)

        #mean(ux), var(ux), mean(uy), var(uy), mean(v), var(v)
        stats = np.zeros((num_incl, 6))

        for i in range(num_incl):

            figname = fname + '-vel-' + str(i + 1)
            plot_hist(vel[incl_ind == (i + 1)], title='', bins=bins,
                      density=True, showfig=showfig, savefig=savefig,
                      savedata=savedata, figname=figname, figformat='pdf')

            stats[i, 0] = np.mean(uxm[incl_ind == (i + 1)])
            stats[i, 1] = np.var(uxm[incl_ind == (i + 1)])
            stats[i, 2] = np.mean(uym[incl_ind == (i + 1)])
            stats[i, 3] = np.var(uym[incl_ind == (i + 1)])
            stats[i, 4] = np.mean(vel[incl_ind == (i + 1)])
            stats[i, 5] = np.var(vel[incl_ind == (i + 1)])


        figname = fname + '-vel-mean'
        plot_hist(stats[:, 4], title='', bins=bins, density=True,
                  showfig=showfig, savefig=savefig, savedata=savedata,
                  figname=figname, figformat='pdf')

        if savedata:
            header = 'mean-ux var-ux mean-uy var-uy mean-v var-v'
            filename = fname + '-vel-incl-stats.dat'
            np.savetxt(filename, stats, header=header)

    return True
################
def velocity_distribution_from_file(fname, folder='.',savedata=True,
                                    directSolver=True):
    ''' Loads data and computes the velocity distributions.
    '''
    permfile = folder + '/' + fname

    grid, circles, Kfactor, Kdist, Kincl = load_perm(permfile)
    #radius = circles[0]['r']
    kperm, incl_ind, Kincl = perm_matrix(grid, circles, Kfactor, Kdist=Kdist, Kincl=Kincl)

    velocity_distribution(grid, kperm, ux=None, uy=None, incl_ind=incl_ind,
                          bins='auto', showfig=False, savefig=False,
                          savedata=savedata, fname=fname,
                          directSolver=directSolver)

    return True
################
def transport_pollock(grid, incl_ind, Npart, ux, uy, isPeriodic=False,
                      plotit=False, CC=None, xcp=None, fname=None,InjSize=1.):
    '''...'''

    # Geometry
    Lx, Ly, Nx, Ny = unpack_grid(grid)

    dx = np.float(Lx/Nx)
    dy = np.float(Ly/Ny)

    # Interpolant
    Ax = ((ux[:, 1:Nx + 1] - ux[:, 0:Nx])/dx).flatten(order='F')
    Ay = ((uy[1:Ny + 1, :] - uy[0:Ny, :])/dy).flatten(order='F')

    #Faces velocities.
    ux1 = ux[:, 0:Nx].flatten(order='F')
    ux2 = ux[:, 1:Nx + 1].flatten(order='F')

    uy1 = uy[0:Ny, :].flatten(order='F')
    uy2 = uy[1:Ny + 1, :].flatten(order='F')

    #faces' coordinates
    x1 = np.arange(0., Lx + dx, dx)
    y1 = np.arange(0., Ly + dy, dy)

    # Case for Pollock's method.
    case_x = pollock_case(ux1, ux2)
    case_y = pollock_case(uy1, uy2)

    #control planes
    num_control_planes = 0
    cp_writen = True

    if xcp is not None:
        num_control_planes = xcp.shape[0]
        cp_writen = np.zeros(xcp.shape)
        arrival_times_cp = np.zeros((Npart, num_control_planes))
        xpmax_cp = np.zeros(num_control_planes)
        xpmin_cp = np.zeros(num_control_planes)

    arrival_times = np.zeros(Npart)

    #number of inclusions
    num_incl = incl_ind.max().astype(int)


    # Time of each particle in each inclusion
    # It contains nincl dictionaries.
    # Each dictionary inclusion contain temporarily the
    # times at which the particles enters and leaves the inclusion.
    # At the end of the simulation only the time spent is stored.
    # Example:
    #
    #   t_in_incl[1] --> times for particles that visited inclusion 1.
    #   t_in_incl[1] = {1: [0.1, 3.4], 5: [1.2, 7.9]}
    #
    #   This means particle 1 entered at time 0.1 and left at time 3.4
    #   and particle 3 entered at 1.2 and left at 7.9
    #
    #   At the end of the simulation we will have
    #
    #   t_in_incl[1] = {1: 3.3, 5: 6.7}
    #
    t_in_incl = []
    t_in_incl_cp = []
    for i in range(num_incl):
        t_in_incl.append({})

    for icp in range(num_control_planes):
        t_in_incl_cp.append([])
        for i in range(num_incl):
            t_in_incl_cp[icp].append({})

    #initial position of particles
    xp = np.zeros(Npart)
    yp = np.arange((InjSize*Ly)/Npart/2.0, InjSize*Ly, (InjSize*Ly)/Npart)
    yp = yp + 0.5 - InjSize/2.

    isIn = np.where(xp < Lx)[0]

    # initial cells indexes.
    indx = (xp[isIn]/dx).astype(int)
    indy = (yp[isIn]/dy).astype(int)

    i = 0

    if plotit and  CC is not None:
        figt, axt, cbt = plot2D(grid, CC)
        lint = None
        figt, axt, lint = plotXY(xp[isIn], yp[isIn], figt, axt, lint)
        axt.set_aspect('equal')
        axt.set_xlim([0., Lx])
        axt.set_ylim([0., Ly])
        axt.set_xticks(np.arange(0, Lx, dx))
        axt.set_yticks(np.arange(0, Ly, dy))
        #plt.grid()
        plt.title(i)

    while  isIn.size > 0:

        i = i + 1

        #Ids of cells where particles are.
        ic = indy[isIn] + indx[isIn]*Ny

        t_in_incl = update_time_in_incl(t_in_incl, incl_ind, isIn,
                                        indx[isIn], indy[isIn], arrival_times)
        #particles velocity
        uxp = (ux1[ic] + Ax[ic]*(xp[isIn] - x1[indx[isIn]]))
        uyp = (uy1[ic] + Ay[ic]*(yp[isIn] - y1[indy[isIn]]))


        #potential travel times and exit faces
        tx = travel_time(case_x[ic], xp[isIn], uxp,
                         ux1[ic], ux2[ic], Ax[ic], dx)

        face_x = exit_face(case_x[ic], uxp, is_y=False)

        ty = travel_time(case_y[ic], yp[isIn], uyp,
                         uy1[ic], uy2[ic], Ay[ic], dy)

        face_y = exit_face(case_y[ic], uyp, is_y=True)

        # actual travel time is the minimum.
        # Some cells give nan but this should mean that there is no exit.
        #qq time = np.array([tx, ty]).min(0)
        time = np.nanmin(np.array([tx, ty]), 0)

        #actual exit face is the corresponding to the minimum travel time.
        ##mm = np.array([tx, ty]).argmin(0)
        mm = np.nanargmin(np.array([tx, ty]), 0)
        
        face = np.array([face_x, face_y])[mm, np.arange(mm.shape[0])]

        # Calculates exit point.
        xp[isIn], yp[isIn] = exit_point(case_x, case_y, ux1, uy1,
                                        xp[isIn], yp[isIn],
                                        uxp, uyp, Ax, Ay,
                                        time, face, dx, dy, ic,
                                        x1, y1, indx[isIn], indy[isIn])
        if i%(1000) == 0:
            print("Last particle at: %e" %(np.min(xp)))
            print("Particles inside: %e" %(isIn.size))

            if plotit:
                figt, axt, lint = plotXY(xp[isIn], yp[isIn], figt, axt, lint)
                axt.set_aspect('equal')
                axt.set_xlim([0., Lx])
                axt.set_ylim([0., Ly])
                axt.set_xticks(np.arange(0, Lx, dx))
                axt.set_yticks(np.arange(0, Ly, dy))
                plt.title(i)

        #arrival times for cbtc.
        arrival_times[isIn] = arrival_times[isIn] + time

        t_in_incl = update_time_in_incl(t_in_incl, incl_ind, isIn,
                                        indx[isIn], indy[isIn], arrival_times)
        # Determine new cell.
        # if exits through face
        #   1 --> indx - 1
        #   2 --> indx + 1
        #  -1 --> indy - 1
        #  -2 --> indy + 1

        indx[isIn] = indx[isIn] + ((face > 0)*(2*face - 3)).astype(int)
        indy[isIn] = indy[isIn] + ((face < 0)*(-2*face - 3)).astype(int)


        #boundary conditions
        if isPeriodic:
            yp[indy > Ny - 1] = 0.
            yp[indy < 0] = Ly
            indy[indy > Ny - 1] = 0
            indy[indy < 0] = Ny - 1

        else:
            yp[indy > Ny - 1]  = Ly
            indy[indy > Ny - 1] = Ny - 1

            yp[indy > Ny - 1] = 0.
            indy[indy < 0] = 0

        #update number of particles still inside the domain.
        isIn = np.where((Lx - xp) > dx/2.)[0]

        #btc, cbtc, and trapping events at intermediate control planes
        # trapping events TO DO  qq
        if num_control_planes>0:
            for icp in range(num_control_planes):
                vorcp = np.where((xcp[icp] - xp) > dx/2.)[0]

                if vorcp.size > 0 and not cp_writen[icp]:
                    #xpmax_cp[icp] = xp[vorcp].max()
                    #xpmin_cp[icp] = xp[vorcp].min()
                    #print('xpmax added ' + str(xpmax_cp[icp]) + ' at plane ' + str(icp)
                    #      + ' x = ' + str(xcp[icp]))
                    #if xpmax_cp[icp] > xcp[icp]:
                    #    ipdb.set_trace()

                    arrival_times_cp[vorcp,icp] = arrival_times[vorcp]
                    t_in_incl_cp[icp] = update_time_in_incl(t_in_incl_cp[icp],incl_ind,
                                                            vorcp, indx[vorcp], indy[vorcp],
                                                            arrival_times_cp[:,icp])
                if vorcp.size < 1 and not cp_writen[icp]:
                    filename = 'cp' + str(icp)

                    if fname is not None:
                       filename = fname + '-' + filename
                    #print('')
                    print('Control plane ' + str(icp) + ' at x = ' + str(xcp[icp]))
                    #print('xpmax ' + str(xpmax_cp[icp]))
                    #print('xpmin ' + str(xpmin_cp[icp]))
                    #print('')

                    _, _ = compute_cbtc(arrival_times_cp[:,icp], bins='auto',
                                        saveit=True, filename=filename)
                    _, _ = compute_btc(arrival_times_cp[:,icp], bins='auto',
                                       saveit=True, filename=filename,showfig=False)
                    _ = inclusion_per_particle(t_in_incl_cp[icp], Npart, saveit=True,
                                               filename=filename)
                    _, _ = time_per_inclusion(t_in_incl_cp[icp], Npart, bins='auto',
                                              saveit=True, filename=filename)

                    np.save(filename + '.npy', arrival_times_cp[:,icp])
                    np.save(filename + '-trap-events.npy', t_in_incl_cp[icp])
                    cp_writen[icp] = True

    return  arrival_times, t_in_incl

#############
def pollock_case(u1, u2):
    ''' Determines to which case in Pollock's (1988) method the velocity
        configuration belongs.
    '''

    case = np.zeros(u1.shape)

    case[(u1 >= 0) & (u2 >= 0)] = 1
    case[(u1 <= 0) & (u2 <= 0)] = 2
    case[(u1 >= 0) & (u2 >= 0) & (np.abs(u1 - u2) < 1e-12)] = 3
    case[(u1 <= 0) & (u2 <= 0) & (np.abs(u1 - u2) < 1e-12)] = 4
    case[(u1 > 0) & (u2 < 0)] = 5
    case[(u1 < 0) & (u2 > 0)] = 6
    
    return case
#############
def travel_time(case, p, up, u1, u2, A, dp):
    ''' Determines the travel time based on the Pollock case.'''

    np.seterr(all='ignore')

    time = 1e19*np.ones(case.shape)

    time = np.where((case == 1) & (u2 !=0.), (1./A)*np.log(u2/up), time)

    time = np.where((case == 2) & (u1 !=0.), (1./A)*np.log(u1/up), time)

    time = np.where(case == 3, dp/u1, time)

    time = np.where(case == 4, dp/u2, time)
    time = np.where(case == 5, 1e19, time)

    time = np.where((case == 6) & (up > 0), (1./A)*np.log(u2/up), time)

    time = np.where((case == 6) & (up < 0), (1./A)*np.log(u1/up), time)

    time[time<0] = 1e19

    return time
#############
def exit_face(case, up, is_y = False):
    ''' Determines the exit face based on the Pollock case.'''

    face = np.zeros(case.shape)

    face = np.where(case == 1, 2, face)

    face = np.where(case == 2, 1, face)

    face = np.where(case == 3, 2, face)

    face = np.where(case == 4, 1, face)

    face = np.where(case == 5, 0, face)

    face = np.where((case == 6) & (up > 0), 2, face)

    face = np.where((case == 6) & (up < 0), 1, face)


    return face*(1 - 2*is_y) #y faces have negative indexes
################
def exit_point(case_x, case_y, ux1, uy1, xp, yp, uxp, uyp, Ax, Ay,
               time, face, dx, dy, ic, x1, y1, indx, indy):

    xpnew = np.where((case_x[ic] == 3) | (case_x[ic] == 4),
                  xp + uxp*time,
                  x1[indx] + (1./Ax[ic])*(uxp*np.exp(Ax[ic]*time) - ux1[ic])
    )

    ypnew = np.where((case_y[ic] == 3) | (case_y[ic] == 4),
                  yp + uyp*time,
                  y1[indy] + (1./Ay[ic])*(uyp*np.exp(Ay[ic]*time) - uy1[ic])
                  )

    xp[face == 1] = x1[indx - 1][face == 1]
    xp[face == 2] = x1[indx + 1][face == 2]
    xp[(face != 1) & (face != 2)] = xpnew[(face != 1) & (face != 2)]

    yp[face == -1] = y1[indy][face == -1]
    yp[face == -2] = y1[indy + 1][face == -2]
    yp[(face != -1) & (face != -2) ] = ypnew[(face != -1) & (face != -2)]

    return xp, yp
##################
def compute_btc(arrival_times, bins='auto', saveit=False,
                 logx=False, logy=False, showfig=False,
                 savefig=False, filename=None, ww=None):
    ''' Breakthrough curve from arrival times.'''

    vals, edges = np.histogram(arrival_times, bins=bins, density=True)
    btc_time = (edges[:-1] + edges[1:])/2.

    if saveit:
        fname = 'btc-h' + '.dat'
        if filename is not None:
            fname = filename + '-' + fname

        np.savetxt(fname, np.matrix([btc_time, vals]).transpose())
    if showfig:
        plotXY(btc_time, vals, logx=logx, logy=logy, allowClose=True)

        if savefig:
            save_fig(xlabel='time', ylabel='cbtc', title='',
                     figname=fname, fogformat='pdf')

    return btc_time, vals
##################
def control_planes_position(Lx, displacement=0.0, control_planes=None):
    ''' Position is calculated from the end of the left buffer.'''

    if control_planes is not None:

        if np.isscalar(control_planes) and control_planes > 0:
            control_planes = np.arange(Lx/control_planes,
                             Lx + Lx/control_planes,
                             Lx/control_planes)

        else:
            control_planes = np.array(control_planes)
            control_planes.sort()

        if (control_planes > 0.).all() and (control_planes <= Lx).all():
            control_planes = control_planes + 0.5*displacement

            print()
            print("Control planes")
            print("======= ======")
            print(control_planes)
            print()

        else:
            print('Wrong control planes!')
            control_planes = None

    return control_planes
##################
def trunc_gamma(shape, theta, a1, a2, size=1):
    ''' Generates random values from a truncated gamma distribution.
          a1 = value for left truncation
          a2 = value for right truncation

        Translated to python from
        https://github.com/ericyewang/GEODE/blob/master/matlab%20code%201.2/gamrndtruncated.m#L10
     '''

    from scipy.stats import gamma
    from scipy.stats import invgamma

    epsilon = 1e-8

    cdf1 = gamma.cdf(a1, shape,scale=1./theta)
    cdf2 = gamma.cdf(a2, shape,scale=1./theta)

    cond1 = (cdf1 > 1. - epsilon) and (cdf2 > 1. - epsilon)
    cond2 = (cdf1 < epsilon) and (cdf2 < epsilon)

    vals =np.full((size,), a1*cond1 + a2*cond2)

    if (not cond1) and (not cond2):
        vals = gamma.ppf(cdf1 + np.random.uniform(0., 1., size)*(cdf2 - cdf1),
                        shape, scale=1./theta)

    return vals
##################
def save_perm(grid, circles, Kfactor, Kdist, Kincl, filename=None):
    ''' Save permeability data'''

    if filename is None:
        fname = 'perm.plk'
    else:
        fname = filename + '-perm.plk'

    with open(fname, 'wb') as ff:
        pickle.dump([grid, circles, Kfactor, Kdist, Kincl],
                    ff, pickle.HIGHEST_PROTOCOL)

##################
def save_react(Rfactor, Rdist, react_rate, filename=None):
    ''' Save permeability data'''

    if filename is None:
        fname = 'react.plk'
    else:
        fname = filename + '-react.plk'

    with open(fname, 'wb') as ff:
        pickle.dump([Rfactor, Rdist, react_rate],
                    ff, pickle.HIGHEST_PROTOCOL)

##################
def chek_overlap(circles):

    r2 = 2.*circles[0]['r']
    i = 0
    print('r2: ', r2)
    for c1 in circles:
        x1 = c1['x']
        y1 = c1['y']
        j = 0
        for c2 in circles:
            x2 = c2['x']
            y2 = c2['y']
            dx2 = (x1 - c2['x'])**2
            dy2 = (y1 - c2['y'])**2
            distance = np.sqrt(dx2 + dy2)
            j = j + 1
            if distance > r2:
                
                i = i + 1
    print('Total overlaps: ', i)
    return

##################    
def particle_velocity_distribution(t_in_incl, incl_ind, ux=None, uy=None, bins='auto',
                                   showfig=False, savefig=False, savedata=True, fname=''):
    '''Given the dictionary the dictionary with the time each particle spent in each inclusion 
       and the mean velocity in the inclusion, returns the inclusions velocity distribution seen
       by the particles.

    # t_in_incl  contains nincl dictionaries. Each dictionary contains the particle and the time spent.
    '''

    if ux is None or uy is None:
        ux, uy = flow(grid, 1./kperm, 'flow', isPeriodic=True, plotHead=False,
                      directSolver=directSolver, tol=tol, maxiter=maxiter)

    #uxm = (ux[:, 0:-1] + ux[:, 1:])/2.
    #uym = (uy[0:-1, :] + uy[1:, :])/2.

    vel = np.sqrt(((ux[:, 0:-1] + ux[:, 1:])/2.)**2 + ((uy[0:-1, :] + uy[1:, :])/2.)**2)

    nincl = len(t_in_incl)

    vmean = np.empty(nincl)

    incl_ind = incl_ind.todense().astype(int)
   
    for i in range(nincl):
        vmean[i] = vel[incl_ind == (i + 1)].mean()
    
    vels = []
    for incl in range(nincl):
        vels.extend(vmean[incl] for i in range(len(t_in_incl[incl])))


    np.savez_compressed(fname + '-vel-part.npz', vels=vels)

    figname = fname + '-vel-part'
    plot_hist(np.array(vels), title='', bins=bins,
              showfig=showfig, savefig=savefig,
              savedata=savedata, figname=figname)

    figname = fname + '-vel-part-log'
    plot_hist(np.array(np.log(vels)), title='', bins=bins,
              showfig=showfig, savefig=savefig,
              savedata=savedata, figname=figname)

    return True

##################
def residence_times(t_in_incl, Npart):
    '''Residence times of particles in each inclusion'''

        
    # total time of each particle in each inclusion given by inclusion
    # total_time has nincl dictionaries.
    # (list of dictionaries with repeated keys)
    total_time = total_time_in_incl(t_in_incl)
        
    # We get the index of trapped particles from dict keys.
    all_keys = set().union(*(d.keys() for d in total_time))
    trapped_part = np.array(list(all_keys), dtype='int')
    #trapped_part = np.fromiter(all_keys, int, len(all_keys))

    # Time spent in inclusions grouped by particle.
    res_times = {k: [d.get(k) for d in total_time if k in d] 
                 for k in set().union(*total_time)}

    
    # Visited inclusions
    visited_incl = {k: [idx for idx, d in enumerate(total_time) if k in d]
                  for k in set().union(*total_time)}

    return res_times, trapped_part, visited_incl

##################
def reactive_arrival_times(arrival_times, t_in_incl, react_rate,
                           saveit=False, filename=None):
    ''' Arrival times with reactions'''

    Npart = arrival_times.shape[0]
    n_incl = len(t_in_incl)

    res_times, trapped_part, visited_incl = residence_times(t_in_incl, Npart)
    # We check if particle reacted
    reacted = []
    
    for particle in trapped_part:
        
        times = np.array(list(res_times.get(particle)))
        rr = react_rate[visited_incl[particle]].reshape(-1,1) #convert to column
        
        react_prob = 1. - np.exp(-rr*times) 
        check = np.random.uniform(size=times.shape)
        if np.any(check < react_prob):
            reacted.append(particle)

    # We remove the arrival times of particles that reacted
    reactive_at = np.delete(arrival_times, reacted)

    if saveit:
        if saveit:
            f = filename + '-react-arrival-times.npy'
            np.save(f, reactive_at)
        
    return reactive_at

##################
def reactive_btcs(r_arrival_times, Npart,
                 bins='auto', saveit=False,
                 logx=False, logy=False, showfig=False,
                 savefig=False, filename=None):
    ''' Breakthrough curve from arrival times with reactions'''

    if saveit:
        filename = filename + '-react'

    
    #Compute curves.
    cbtc_time, cbtc = compute_cbtc(r_arrival_times, Npart=Npart, bins=bins,
                                   saveit=saveit, showfig=showfig,
                                   savefig=savefig, filename=filename)

    btc_time, btc = compute_btc(r_arrival_times,  bins=bins, saveit=saveit,
                                showfig=showfig, savefig=savefig,
                                filename=filename)

    #rescale btc
    btc = btc *r_arrival_times.shape[0]/Npart
    
    return btc_time, btc, cbtc_time, cbtc 

##################
def calc_distributed_perm(Kfactor, Kdist, n_incl):
    ''' Computes permeability distribution.'''

    if Kdist=='uni':
            Kincl = np.random.uniform(Kfactor.min(), Kfactor.max(), size=n_incl)
    elif Kdist=='lognorm':
        #Kfactor[0] -> Geometric mean of K.
        #Kfactor[1] -> Variance of logK.
        Kincl = np.random.normal(0., 1.,size=n_incl)
        Kincl = Kfactor[0]*np.exp(Kincl*np.sqrt(Kfactor[1]))

    elif Kdist=='tgamma':
        # a -> shape.
        # b -> scale.
        #mean = b*a;  variance=a*b^2
        #Kfactor[0] -> shape as in wikipedia (k)
        #Kfactor[1] -> scale as in wikipedia (theta). Scipy calls scale to the inverse.
        #Kfactor[2] -> Left truncation value.
        #Kfactor[3] -> Right truncation value.
        Kincl = trunc_gamma(Kfactor[0], Kfactor[1], Kfactor[2], Kfactor[3], size=n_incl)
    elif Kdist=='gamma':
        # a -> shape.
        # b -> scale.
        #mean = b*a;  variance=a*b^2
        #Kfactor[0] -> shape as in wikipedia (k)
        #Kfactor[1] -> scale as in wikipedia (theta).
        Kincl =  np.random.gamma(Kfactor[0], scale=Kfactor[1], size=n_incl)
        
    else: # default constant
        Kincl = np.full((n_incl,), Kfactor[0])

    return Kincl

##################
def comp_react_rate(Rfactor, Rdist, n_incl, Kincl, Kfactor, saveit=False, filename=None):
    ''''Computes reaction rates given the describing parameters.'''

    # Compute reaction rate per particle.
    Rfactor = np.atleast_1d(Rfactor)
    if Rdist=='correlated':
        #Rfactor[0] -> Geometric mean of R.
        #Rfactor[1] -> Variance of logR.
        #Rfactor[2] -> correlation coefficiente
        #Kfactor[0] -> Geometric mean of K.
        #Kfactor[1] -> Variance of logK.
        react_rate = correlated_var(Kincl, Rfactor[2], undo_transform=True, Xmean=Kfactor[0], Xvar=Kfactor[1])
        react_rate = Rfactor[0]*np.exp(react_rate*np.sqrt(Rfactor[1]))

    else:
        react_rate = calc_distributed_perm(Rfactor, Rdist, n_incl)

    if saveit:
        save_react(Rfactor, Rdist, react_rate, filename=filename)

    return react_rate

##################
def correlated_var(X1, corr_coef, undo_transform=False, Xmean=None, Xvar=None):
    '''Generates a variable correlated to a given one using Cholesky transformation.
       From:
https://stats.stackexchange.com/questions/450771/cholesky-decomposition-or-alternative-for-negatively-correlated-data-simulations'''

    #Undoes the tranformation so that we get mean=0 and var=1
    if undo_transform and Xmean is not None and Xvar is not None:
        X1 = (1./np.sqrt(Xvar))*np.log(X1/Xmean)

    # Correlation matrix
    corr_mat = corr_coef*np.ones((2,2))
    np.fill_diagonal(corr_mat, 1.)

    Lmat = np.linalg.cholesky(corr_mat)

    # New variable
    X2 = np.random.normal(0., 1.,size=X1.shape[0])

    # We construct the correlated variable
    data = np.vstack((X1, X2))
    XX = Lmat.dot(data)

    return XX[1, :]
