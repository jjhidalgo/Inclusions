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
################
def run_simulation(*, Lx=1., Ny=50,
                   pack='tri', n_incl_y=3, Kfactor=0.1,
                   bcc='head', isPeriodic=True, integrateInTime=True,
                   tmax=10., dt=None, Npart=100,
                   plotPerm=False, plotFlow=False,
                   plotTpt=False, plotBTC=False,
                   filename=None, doPost=True):
    """ Runs a simulation."""

    grid = setup_grid(Lx, Ny)

    kperm, incl_ind, grid = permeability(grid, n_incl_y, Kfactor,
                                         pack=pack, filename=filename,
                                         plotit=plotPerm, saveit=True)


    ux, uy = flow(grid, kperm, bcc, isPeriodic=isPeriodic, plotHead=plotFlow)

    if dt is None:
        if integrateInTime:
            tx = grid['Lx']/grid['Nx']/ux.max()
            ty = grid['Ly']/grid['Ny']/uy.max()
            dt = np.min([tx, ty, 1e-3])
        else:
            dt = 0.1*Kfactor*grid['Lx']/grid['Nx']


    if integrateInTime:
        arrival_times, t_in_incl = transport(grid, incl_ind,
                                             Npart, ux, uy,
                                             tmax, dt, isPeriodic=isPeriodic,
                                             plotit=plotTpt, CC=kperm)
    else:
        arrival_times, t_in_incl = transport_ds(grid, incl_ind,
                                                Npart, ux, uy,
                                                dt, isPeriodic=isPeriodic)


    if filename is None:
        filename = 'K' + str(Kfactor).replace('.', '') + pack + 'Ninc' + str(n_incl_y)

    cbtc_time, cbtc = compute_cbtc(arrival_times,
                                   saveit=True, filename=filename)

    np.savetxt(filename + '-btc.dat', np.matrix([cbtc_time, cbtc]).transpose())

    with open(filename + '.plk', 'wb') as ff:
        pickle.dump([Npart, t_in_incl, arrival_times], ff, pickle.HIGHEST_PROTOCOL)

    if plotBTC:
        _, _, _ = plotXY(cbtc_time, 1. - cbtc, allowClose=True)
    print("End of simulation.\n")

    if doPost:

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
    return grid['Lx'][0], grid['Ly'][0], grid['Nx'][0], grid['Ny'][0]

################
def permeability(grid, n_incl_y, Kfactor=1., pack='sqr', filename=None, plotit=False, saveit=True):
    """Computes permeability parttern inside a 1. by Lx rectangle.
       The area covered by the inclusions is 1/2 of the rectanle.
       If the arrangement os random, the radius is reduced by 10%
       to increase the chance of achieving the target porosity.

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

    radius = np.sqrt(np.float(Lx)/(2. * np.pi * np.float(n_incl)))

    if pack == 'sqr' or pack == 'tri':
        pore = rp.RegPore2D(nx=n_incl_x, ny=n_incl_y,
                            radius=radius, packing=pack)
        throat = (Ly - 2.0*np.float(n_incl_y)*radius)/(np.float(n_incl_y) + 1.0)
        pore.throat = throat
        pore.bounding_box = ([0.0, 0.0, 0.5], [Lx, Ly, 1.0])
        pore.xoffset = 0.


        #delta = Lx - n_incl_x*(throat + 2.*radius)
        #displacement = (delta - throat)/2.

    elif pack == 'rnd':

        pore = rp.RndPore2D(lx=Lx, ly=Ly,
                            rmin=0.90*radius, rmax=0.90*radius,
                            target_porosity=0.6, packing='rnd')

        pore.ngrains_max = int(1.1*n_incl)
        pore.ntries_max = int(1e5)


    # Resizes domain to avoid boundary effects
    # (2*radius added to the left and to the right).
    displacement = np.ceil(4.*radius)
    pore.circles[:]['x'] = pore.circles[:]['x'] + 0.5*displacement

    # if no discretization is given a 30th of the smallest inclusion's
    # radius is chosen as cell size.
    if Ny < 1:
        Ny = np.int(30/pore.circles[:]['r'].min())

    grid = setup_grid(Lx + displacement, Ny)

    kperm, incl_ind = perm_matrix(grid, pore.circles, Kfactor)

    # kperm[xx>-1] = 1.
    # kperm[yy>0.5] = 0.1
    # kperm[xx<0.25] = 1.
    # kperm[xx>0.75] = 1.
    if plotit:
        plot2D(grid, kperm, title='kperm', allowClose=True)

    if saveit:
        if filename is None:
            fname = 'perm.plk'
        else:
            fname = filename + '-perm.plk'

        with open(fname, 'wb') as ff:
            pickle.dump([grid, pore.circles, Kfactor],
                        ff, pickle.HIGHEST_PROTOCOL)

    return kperm, incl_ind, grid

################
def flow(grid, kperm, bcc, isPeriodic=True, plotHead=False):
    ''' Solves the flow equation and returns the velocity
        at the cell's faces.'''

    Lx, Ly, Nx, Ny = unpack_grid(grid)

    dx = Lx/Nx
    dy = Ly/Ny
    dx2 = dx*dx
    dy2 = dy*dy
    mu = 1./kperm
    Np = Nx*Ny

    # Transmisibility matrices.

    Tx = np.zeros((Ny, Nx + 1))
    #Tx[:, 1:Nx] = (2.*dy)/(mu[:, 0:Nx-1] + mu[:, 1:Nx+1])
    Tx[:, 1:Nx] = (2.*dy)/(mu[:, 0:Nx-1] + mu[:, 1:Nx]) #new
    Ty = np.zeros([Ny + 1, Nx])
    #Ty[1:Ny, :] = (2.*dx)/(mu[0:Ny-1, :] + mu[1:Ny+1, :])
    Ty[1:Ny, :] = (2.*dx)/(mu[0:Ny-1, :] + mu[1:Ny, :]) #new

    Tx1 = Tx[:, 0:Nx].reshape(Np, order='F')
    Tx2 = Tx[:, 1:Nx+1].reshape(Np, order='F')

    Ty1 = Ty[0:Ny, :].reshape(Np, order='F')
    Ty2 = Ty[1:Ny+1, :].reshape(Np, order='F')

    Ty11 = Ty1
    Ty22 = Ty2

    TxDirich = np.zeros(Ny)
    TxDirich = (2.*dy)*(1./mu[:, Nx-1])


    if bcc == 'head':
        TxDirichL = (2.*dy)*(1./mu[:, 0])

    #Assemble system of equations
    Dp = np.zeros(Np)
    Dp = Tx1/dx2 + Ty1/dy2 + Tx2/dx2 + Ty2/dy2

    #Dirichlet b.c. on the right
    Dp[Np-Ny:Np] = Ty1[Np-Ny:Np]/dy2 + Tx1[Np-Ny:Np]/dx2 + \
                   Ty2[Np-Ny:Np]/dx2 + TxDirich/dx2

    if bcc == 'head':
        Dp[0:Ny] = Ty1[0:Ny]/dy2 + Tx2[0:Ny]/dx2 + \
                   Ty2[0:Ny]/dx2 + TxDirichL/dx2

    #Periodic boundary conditions
    TypUp = np.zeros(Np)
    TypDw = np.zeros(Np)

    if isPeriodic:
        Typ = (2.*dx)/(mu[Ny - 1, :] + mu[0, :])

        TypDw[0:Np:Ny] = Typ
        TypUp[Ny-1:Np:Ny] = Typ

        Dp[0:Np:Ny] = Dp[0:Np:Ny] + Typ/dy2
        Dp[Ny-1:Np:Ny] = Dp[Ny-1:Np:Ny] + Typ/dy2

    Am = sp.spdiags([-Tx2/dx2, -TypDw/dy2, -Ty22/dy2, Dp,
                     -Ty11/dy2, -TypUp/dy2, -Tx1/dx2],
                    [-Ny, -Ny + 1, -1, 0, 1, Ny - 1, Ny], Np, Np,
                    format='csr')


    #RHS - Boundary conditions
    u0 = 1.*dy
    hL = 1.
    hR = 0.
    S = np.zeros(Np)
    S[Np-Ny:Np+1] = TxDirich*(hR/dx/dx) # Dirichlet X=1;
    if bcc == 'head':
        S[0:Ny] = TxDirichL*(hL/dx/dx) # Dirichlet X=0;
    else:
        S[0:Ny] = u0/dx #Neuman BC x=0;

    head = lgsp.spsolve(Am, S).reshape(Ny, Nx, order='F')

    #Compute velocities

    ux = np.zeros([Ny, Nx+1])
    uy = np.zeros([Ny+1, Nx])

    if bcc == 'head':
        ux[:, 0] = -TxDirichL*(head[:, 0] - hL)/dx
    else:
        ux[:, 0] = u0

    #ux[:, 1:Nx] = -Tx[:, 1:Nx]*(head[:, 1:Nx+1] - head[:, 0:Nx-1])/dx
    ux[:, 1:Nx] = -Tx[:, 1:Nx]*(head[:, 1:Nx] - head[:, 0:Nx-1])/dx
    ux[:, Nx] = -TxDirich*(hR - head[:, Nx-1])/dx

    #uy[1:Ny, :] = -Ty[1:Ny, :]*(head[1:Ny+1, :] - head[0:Ny-1, :])/dy
    uy[1:Ny, :] = -Ty[1:Ny, :]*(head[1:Ny, :] - head[0:Ny-1, :])/dy

    #periodic
    if isPeriodic:
        uy[0, :] = Typ*(head[0, :] - head[Ny - 1, :])/dy
        uy[Ny, :] = uy[0, :]


    if plotHead:
        plot2D(grid, head, title='head', allowClose=True)
        plot2D(grid, ux/dy, title='ux', allowClose=True)
        plot2D(grid, uy/dx, title='uy', allowClose=True)

    return ux/dy, uy/dx

#####
def transport(grid, incl_ind, Npart, ux, uy, tmax, dt, isPeriodic=False,
              plotit=False, CC=None):
    '''Solves the transport of a line of concentration initially at the
       left boundary using a particle tracking method.

       Returns the arrival times of the particles to the right boundary
       and data about the time spent in the inclusions.'''

    Lx, Ly, Nx, Ny = unpack_grid(grid)

    if plotit and  CC is not None:
        figt, axt, cbt = plot2D(grid, CC)

    t = 0.
    xp = np.zeros(Npart)
    #yp = np.random.rand(Npart)##
    yp = np.arange(Ly/Npart/2.0, Ly, Ly/Npart)
    dx = np.float(Lx/Nx)
    dy = np.float(Ly/Ny)

    Ax = ((ux[:, 1:Nx + 1] - ux[:, 0:Nx])/dx).flatten(order='F')
    Ay = ((uy[1:Ny + 1, :] - uy[0:Ny, :])/dy).flatten(order='F')

    ux = ux[:, 0:Nx].flatten(order='F')
    uy = uy[0:Ny, :].flatten(order='F')

    x1 = np.arange(0., Lx + dx, dx) #faces' coordinates
    y1 = np.arange(0., Ly + dy, dy)

    i = 0

    lint = None

    #number of inclusions
    num_incl = incl_ind.max().astype(int)


    #time of each partile in each inclusion
    # It contains nincl dictionaries
    # Each dictionary contains the particle and the time spent.
    t_in_incl = []

    for i in range(num_incl):
        t_in_incl.append({})

    nwrite = 10# np.max([int((tmax/dt)/1000),10])
    arrival_times = np.zeros(Npart)
    i = 0

    isIn = np.where(xp < Lx)[0]

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
                figt, axt, lint = plotXY(xp[isIn], yp[isIn], figt, axt, lint)
                axt.set_aspect('equal')
                axt.set_xlim([0., Lx])
                axt.set_ylim([0., Ly])
                plt.title(t)

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
    #yp = np.random.rand(Npart)
    dx = np.float(Lx/Nx)
    dy = np.float(Ly/Ny)

    #qq
    # ux = np.ones(ux.shape)
    # uy = np.zeros(uy.shape)
    # Npart = 10
    # xp = np.arange(0,1,1/Npart)
    # yp = np.random.rand(Npart)
    #qq

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
def plotXY(x, y, fig=None, ax=None, lin=None, allowClose=False):
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

    if allowClose:
        input("Dale enter y cierro...")
        plt.close()
        fig = None
        ax = None
        lin = None
    return fig, ax, lin
####
def plot2D(grid, C, fig=None, ax=None, title=None, allowClose=False):
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
        y1 = np.arange(0., Ly + dy, dy)

    elif cny < Ny: # y face centered

        x1 = np.arange(0., Lx + dx, dx)
        y1 = np.arange(-dy/2., Ly + dy, dy)

    else:              #cell centered

        x1 = np.arange(0., Lx + dx/2., dx)
        y1 = np.arange(0., Ly + dy/2., dy)

    xx, yy = np.meshgrid(x1, y1)

    plt.sca(ax)
    plt.ion()

    mesh = ax.pcolormesh(xx, yy, C, cmap='coolwarm')

    #plt.axis('equal')
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

    return fig, ax, cb
################
def number_of_grains(nix, niy, pack):
    """Computes number of grains according to the packing"""

    if pack == 'sqr':
        ngrains = nix*niy
    else:
        if nix%2 == 0:
            ngrains = (nix//2)*(niy - 1) + (nix//2)*niy
        else:
            ngrains = (nix//2)*(niy - 1) + (nix//2 + 1)*(niy)

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
def time_per_inclusion(time_in_incl, saveit=False, filename=None):
    """ Given the dictionary whith the time at which each particle entered
        and exited each inclusions, returns the time each particle spent
        in each inclusion in a matrix format suitable to calculate
        histograms.
        Optionally, the data is saved as a text file.
    """
    #First the total time each particle spent in each inclusion is computed.
    tot_time_in_incl = total_time_in_incl(time_in_incl)

    #dictionary with the time particles spent in each inclusion.
    incl_times = {}
    i = 0
    for incl in tot_time_in_incl:
        incl_times[i] = np.concatenate(list(incl.values()))
        i = i + 1

    if saveit:

        if filename is None:
            fname = 'incl-times.dat'
        else:
            fname = filename + '-incl-times.dat'

        from itertools import zip_longest

        import csv
        with open(fname, 'w') as ff:
            writer = csv.writer(ff, delimiter=' ')
            # I whished I knew why this works...
            for values in zip_longest(*list(incl_times.values())):
                writer.writerow(np.asarray(values))

        if filename is None:
            fname = 'trap-times.dat'
        else:
            fname = filename + '-trap-times.dat'

        trap_times =  np.concatenate(np.array(list(incl_times.values())))
        np.savetxt(fname, trap_times)
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
def plot_hist(data, title='', bins='auto', showfig=True, savefig=False,
              savedata=False, figname='zz', figformat='pdf'):
    '''Plots the histogram of the data.'''

    vals, edges, _ = plt.hist(data, bins=bins, normed=True)
    plt.xlabel('time')
    plt.ylabel('freq')
    plt.title(title)

    if savedata:
        filename = figname + '.dat'
        np.savetxt(filename, np.matrix([edges[:-1],edges[1:], vals]).transpose())
    if savefig:
        if figformat == 'pdf':
            print(figname)
            plt.savefig(figname + '.pdf', format='pdf')
        if figformat == 'tikz':
            from matplotlib2tikz import save as tikz_save
            tikz_save(figname + '.tex')

    if showfig:
        plt.ion()
        plt.show()
        input("Dale enter y cierro...")

    plt.close()
    return True
################
def postprocess_from_file(fname, savedata=True, savefig=False,
                          showfig=False, figformat='pdf',
                          bins='auto',dofullpostp=False):
    """Post process from plk file. Computes the histograms for
       particles and inclusions. Saves data and/or figures.
    """

    Npart, t_in_incl, arrival_times = load_data(fname + '.plk')
    postprocess(Npart, t_in_incl, arrival_times, fname='',
                savedata=savedata, savefig=savefig,
                showfig=showfig, figformat=figformat,
                bins=bins, dofullpostp=dofullpostp)
    return True

################
def postprocess(Npart, t_in_incl, arrival_times, fname='',
                savedata=True, savefig=False,
                showfig=False, figformat='pdf',
                bins='auto', dofullpostp=False):
    """Computes the histograms for
       particles and inclusions. Saves data and/or figures.
    """

    _, t_immobile = mobile_inmmobile_time(t_in_incl, arrival_times,
                                          filename=fname, saveit=savedata)

    incl_times, trap_times = time_per_inclusion(t_in_incl,
                                       saveit=savedata, filename=fname)

    figname = fname + '-trap-dist'
    plot_hist(trap_times, title='', bins='auto',
              showfig=showfig, savefig=savefig,
              savedata=savedata, figname=figname)


    _, _ = incl_per_time(t_in_incl, plotit=showfig,
                             saveit=savedata, filename=fname)

    _, _, _, _ = free_trapped_arrival(arrival_times, t_immobile,
                                         saveit=savedata, filename=fname)

    aa = inclusion_per_particle(t_in_incl, Npart,
                                    saveit=savedata, filename=fname)
    #aa -->   number of inclusions visited by each particle.
    figname = fname + '-trap-events'
    plot_hist(aa, title='', bins='auto',
              showfig=showfig, savefig=savefig, savedata=savedata,
              figname=figname)


    if dofullpostp:
    #particle histogram
        plot_hist(trap_times, title=fname + ' particles', bins=bins,
              showfig=showfig, savefig=savepartfig, savedata=savedata,
              figname=fname + '-part-hist', figformat=figformat)


        inclusions_histograms(incl_times, showfig=showfig, savefig=saveinclfig,
                              savedata=False, fname=fname,
                              figformat=figformat, bins=bins)
    return True
################
def postprocess_all(fname, savedata=True, savefig=False,
                    showfig=False, figformat='pdf',
                    bins='auto',dofullpostp=False):

    """ Post process al the cases in a folder."""

    import os as os

    files = os.listdir()
    for file in files:
        if file.endswith('plk'):
            fname = os.path.splitext(file)[0]
            postprocess_from_file(fname, savedata=savedata, savefig=savefig,
                          showfig=showfig, figformat='pdf',
                                  bins='auto', dofullpostp=dofullpostp)
    return True
################
def stream_function(grid, kperm, isPeriodic=False, plotPsi=False):
    '''Compute the stream function.
    The stream function is prescribed at the boundaries so that
    It is equivalent to prescribed flow on the left and right
    boundaries and no flow on top and bottom boundaries.'''

    Lx, Ly, Nx, Ny = unpack_grid(grid)

    dx = Lx/(Nx-1)
    dy = Ly/(Ny-1)
    Np = Nx*Ny

    D1x, D2x, D1y, D2y = fd_mats(Nx, Ny, dx, dy)

    Y = np.log(kperm).reshape(Np, order='F')
    #Kmat = (D1x.multiply((D1x*Y).reshape(Np, 1)) +
    #        D1y.multiply((D1y*Y).reshape(Np, 1)))

    Amat = (D2y + D2x - (D1x.multiply((D1x*Y).reshape(Np, 1)) +
                         D1y.multiply((D1y*Y).reshape(Np, 1)))).tolil()

    RHS = np.zeros(Np)
    #BC
    #
    if isPeriodic:
        #TO DO
        print('stream_function: periodic b. c. not implemented')
    else:
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
    #Left boundary
    Amat[0:Ny, :] = 0.0
    idx = np.arange(0, Ny, 1)
    Amat[idx, idx] = 1.0
    RHS[0:Ny] = np.arange(0., Ly+dy, dy)[::-1] #(Ly:-dy:0)
    #
    #Right boundary
    Amat[Np-Ny:Np, :] = 0.0
    idx = np.arange(Np-Ny, Np, 1)
    Amat[idx, idx] = 1.0
    RHS[Np-Ny:Np] = np.arange(0., Ly+dy, dy)[::-1] #(Ly:-dy:0)
    
    psi = lgsp.spsolve(Amat.tocsr(), RHS).reshape(Ny, Nx, order='F')

    if plotPsi:
        fig, ax, cb = plot2D(grid, kperm, title='psi', allowClose=False)
        cb.remove()
        x1 = np.arange(0.0, Lx+dx, dx)
        y1 = np.arange(0.0, Ly+dy, dy)
        xx, yy = np.meshgrid(x1, y1)
        ax.contour(xx, yy, psi, 21, linewidths=0.5, colors='y')
        input("Dale enter y cierro...")
        plt.close()
    return psi
################
def fd_mats(Nx, Ny, dx, dy):
    '''Computes finite differeces matrices.'''
    Np = Nx*Ny
    dx2 = dx*dx
    dy2 = dy*dy

    # First derivatives in Y
    Dp = np.zeros(Np)
    Dup = np.ones(Np)/(2.0*dy)
    Ddw = -np.ones(Np)/(2.0*dy)
    #
    #Top boundary
    Dp[0:Np:Ny] = -1.0/dy #diagonal
    Ddw[Ny-1:Np:Ny] = 0.0 #diagonal inferior
    Dup[1:Np:Ny] = 1.0/dy #diagonal superior
    #
    #Bottom bounday
    Dp[Ny-1:Np:Ny] = 1.0/dy #diagonal
    Ddw[Ny-2:Np:Ny] = -1.0/dy #diagonal inferior
    Dup[Ny:Np:Ny] = 0.0
    #
    D1y = sp.spdiags([Ddw, Dp, Dup], [-1, 0, 1], Np, Np, format='csr')
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
    #
    # Top boundary
    Dp[0:Np:Ny] = 1.0/dy2 #diagonal
    Ddw[Ny-1:Np:Ny] = 0. # diagonal inferior
    Dup[1:Np:Ny] = -2.0/dy2 #diagonal superior
    Dup2 = np.zeros(Np) #segunda diagonal superior
    Dup2[2:Np:Ny] = 1.0/dy2
    #
    # Bottom boundary
    Dp[Ny-1:Np:Ny] = 1.0/dy2 #diagonal
    Ddw[Ny-2:Np:Ny] = -2.0/dy2 #diagonal inferior
    Ddw2 = np.zeros(Np) #segunda diagonal inferior
    Ddw2[Ny-3:Np:Ny] = 1.0/dy2
    Dup[Ny:Np:Ny] = 0.0 #Diagonal superior

    D2y = sp.spdiags([Ddw2, Ddw, Dp, Dup, Dup2],
                     [-2, -1, 0, 1, 2],
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
        the cells where the poarticles are (indx, indy), the indexes
        of the cells inside incluisions (incl_ind), and the current time
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

    #Update of time spent by particles in each inclusion
    for part, inc in zip(parts_in_incl, incl):
        try:
            t_in_incl[inc][part][1] = time[part]
        except:
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
    tot_time_in_incl = t_in_incl.copy()

    for dic1, incl in zip(tot_time_in_incl, incl):
        tot_time_in_incl[incl] = {k:np.diff(v) for k, v in dic1.items()}

    return tot_time_in_incl
################
def compute_cbtc(arrival_times, saveit=False, filename=None):
    ''' Cummulative breakthrough curve from arrival times.'''
    cbtc_time = np.sort(arrival_times)
    Npart = arrival_times.shape[0]
    cbtc = np.cumsum(np.ones(Npart))/Npart
    if saveit:
        fname = 'cbtc.dat'
        if filename is not None:
            fname = filename + '-' + fname

        np.savetxt(fname, np.matrix([cbtc_time, cbtc]).transpose())
    return cbtc_time, cbtc

################
def mobile_inmmobile_time(t_in_incl, arrival_times, filename=None, saveit=False):

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
def perm_matrix(grid, circles, Kfactor):
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

    i = 0

    for circ in circles:
        i = i + 1
        x0 = circ['x']
        y0 = circ['y']
        r = circ['r']
        mask = ((xx - x0)**2.0 + (yy - y0)**2.0) < r**2.0
        kperm[mask] = Kfactor
        incl_ind[mask] = i

    return kperm, sp.csr_matrix(incl_ind)
################
def load_perm(filename):
    """ Loads the permeability distribution in the given plk file."""


    with open(filename, 'rb') as ff:
        data = pickle.load(ff)
        grid = data[0]
        circles = data[1]
        Kfactor = data[2]

    return grid, circles, Kfactor
################
def incl_per_time(t_in_incl, plotit=False, saveit=False, filename=None):
    """ Given the dictionary whith the time at which each particle entered
        and exited each inclusion, returns the total inclusions that contain
        at least one particle at a given time.
    """

    # total time of particles in inclusions
    num_incl = len(t_in_incl)
    incl_indx = np.arange(0, num_incl, 1)

    # First we obtain the latest time a prticle exited an inclusion
    #    Explanation:
    #        {k:np.max(v) for k, v in dic1.items()} is a dictionary with
    #        the maximum time per particle in inclusion incl.
    #        with max( {---}.values() we get the maximum of all particles
    #        Then with np.max([tmax, max(...)]) we update the maximum.

    tmax = 0.0
    for dic1, incl in zip(t_in_incl, incl_indx):
        tmax = max([tmax, max({k:np.max(v) for k, v in dic1.items()}.values())])

    # Then we divide the time interval in 1000 parts.
    times = np.arange(0.0, tmax + tmax/1000, tmax/1000)
    occ_incl = np.zeros(times.shape[0])
    i = 0

    #For each time we check how many inclusions are occupied.
    for t in times:
        for dic1, incl in zip(t_in_incl, incl_indx):
            aux = np.asarray(list(dic1.values()))
            occ_incl[i] = occ_incl[i] + np.any(
                (aux[:, 0] <= t) & (aux[:, 1] >= t)
            )

        i = i + 1

    if plotit:
        plotXY(times, occ_incl, allowClose=True)
       # vincl[i] = dd

    if saveit:
        fname = 'occ-incl.dat'

        if filename is not None:
            fname = filename + '-' + fname

        np.savetxt(fname, np.matrix([times, occ_incl]).transpose())

    return times, occ_incl
################
def plot_perm_from_file(filename):
    '''Load permeability data from plk file and plots it.'''
    #TO DO : check that file exists.

    grid, circles, Kfactor = load_perm(filename)
    kperm, _ = perm_matrix(grid, circles, Kfactor)
    plot2D(grid, kperm, title='K', allowClose=True)
################
def free_trapped_arrival(arrival_times, t_immobile, saveit=False,
                         filename=None):
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

    traptime, trapcbtc = compute_cbtc(arrivals_trapped,
                                      saveit=saveit, filename=filename + 'trap')
    
    freetime, freecbtc = compute_cbtc(arrivals_free,
                                      saveit=saveit, filename=filename + 'free')


    return traptime, trapcbtc, freetime, freecbtc
################
def inclusion_per_particle(time_in_incl, Npart, saveit=False, filename=None):
    """ Given the dictionary whith the time at which each particle entered
        and exited each inclusions, returns the number of inclusons visited
        by each particle.
        Optionally, the data is saved as a text file.
    """
    incl_per_part = np.zeros(Npart)
    for incl in time_in_incl:
        incl_per_part[list(incl.keys())] = incl_per_part[list(incl.keys())] + 1

    if saveit:
        fname = 'visited-incl.dat'
        if filename is not None:
            fname = filename + '-' + fname
            
        np.savetxt(fname, incl_per_part)
    return incl_per_part
