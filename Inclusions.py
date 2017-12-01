import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lgsp
#import ipdb
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

    kperm, incl_ind, grid = permeability(grid, n_incl_y, Kfactor, pack, plotPerm)

    ux, uy = flow(grid, kperm, bcc, isPeriodic=isPeriodic, plotHead=plotFlow)

    if dt is None:
        if integrateInTime:
            tx = grid['Lx']/grid['Nx']/ux.max()
            ty = grid['Ly']/grid['Ny']/uy.max()
            dt = np.min([tx, ty, 1e-3])
        else:
            dt = 0.1*Kfactor*grid['Lx']/grid['Nx']


    if integrateInTime:
        arrival_time, t_in_incl = transport(grid, incl_ind,
                                            Npart, ux, uy,
                                            tmax, dt, isPeriodic=isPeriodic,
                                            plotit=plotTpt, CC=kperm)
    else:
        arrival_time, t_in_incl = transport_ds(grid, incl_ind,
                                               Npart, ux, uy,
                                               dt, isPeriodic=isPeriodic)


    if filename is None:
        filename = 'K' + str(Kfactor).replace('.', '') + pack + 'Ninc' + str(n_incl_y)

    cbtc_time, cbtc = compute_cbtc(arrival_time)
    np.savetxt(filename + '-btc.dat', np.matrix([cbtc_time, cbtc]).transpose())

    with open(filename + '.plk', 'wb') as ff:
        pickle.dump([Npart, t_in_incl, arrival_time], ff, pickle.HIGHEST_PROTOCOL)

    if plotBTC:
        _, _, _ = plotXY(cbtc_time, 1. - cbtc,allowClose=True)
    print("End of simulation.\n")

    if doPost:
        _, _ = time_distributions(Npart, t_in_incl, writeit=True, fname=filename)
        print("End of postprocess.\n")

    return True
################
def setup_grid(Lx, Ny):
    """Grid set up."""

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
def permeability(grid, n_incl_y, Kfactor=1., pack='sqr', plotit=False, saveit=False):
    """Computes permeability parttern inside a 1. x Lx rectangle.
       The area covered by the inclusions is 1/2 of the rectanle.
       Then de domain is resized by adding 3*radius on the left
       avoid boundary effects.
       The dimension of the box and the discretization are changed
       so that the grid remains regular.

       The function returns the permeability field, the indexes of
       location fo the inclusions and the new grid.
    """

    import RecPore2D as rp

    Lx, Ly, Nx, Ny = unpack_grid(grid)

    n_incl_x = np.int(Lx*n_incl_y)
    n_incl = number_of_grains(n_incl_y, n_incl_x, pack)

    radius = np.sqrt(np.float(Lx)/(2. * np.pi * np.float(n_incl)))


    if pack=='sqr' or pack=='tri':
        pore = rp. RegPore2D(nx=n_incl_x, ny=n_incl_y, radius=radius, packing=pack)
        throat = (Ly - 2.0*np.float(n_incl_y)*radius)/(np.float(n_incl_y) + 1.0)
        pore.throat = throat
        pore.bounding_box = ([0.0, 0.0, 0.5], [Lx, Ly, 1.0])
        pore.xoffset = 0.


        #delta = Lx - n_incl_x*(throat + 2.*radius)
        #displacement = (delta - throat)/2.

    elif pack=='rnd':

        pore = rp.RndPore2D(lx=Lx, ly=Ly,
                            rmin=0.99*radius, rmax=0.99*radius,
                            target_porosity=0.5, packing='rnd')
        pore.ntries_max = int(1e4)

    #centers the circles in bounding box.
    displacement = np.ceil(3.*radius)
    circles = pore.circles
    circles[:]['x'] = circles[:]['x'] + displacement

    #
    # Resizes domain to avoid boundary effects
    grid = setup_grid(Lx + displacement, Ny)
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

    # kperm[xx>-1] = 1.
    # kperm[yy>0.5] = 0.1
    # kperm[xx<0.25] = 1.
    # kperm[xx>0.75] = 1.

    if plotit:
        plot2D(kperm, Nx, Ny, Lx, Ly, title='kperm', allowClose=True)


    if saveit:
            with open('perm.plk', 'wb') as ff:
                pickle.dump([Nx,Ny,Lx,Ly, kperm], ff, pickle.HIGHEST_PROTOCOL)
    return kperm, sp.csr_matrix(incl_ind), grid

################
def flow(grid, kperm, bcc, isPeriodic=True, plotHead=False):

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
        uy[0, :] =  Typ*(head[0,:] - head[Ny - 1,:])/dy
        uy[Ny, :] =  uy[0,:]


    if plotHead:
        plot2D(head, Nx, Ny, Lx, Ly, title='head', allowClose=True)
        plot2D(ux/dy, Nx, Ny, Lx, Ly, title='ux', allowClose=True)
        plot2D(uy/dx, Nx, Ny, Lx, Ly, title='uy', allowClose=True)

    return ux/dy, uy/dx

#####
def transport(grid, incl_ind, Npart, ux, uy, tmax, dt, isPeriodic=False,
              plotit=False, CC=None):

    Lx, Ly, Nx, Ny = unpack_grid(grid)

    if plotit and  CC is not None:
        figt, axt = plot2D(CC, Nx, Ny, Lx, Ly)

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
    arrival_time = np.zeros(Npart)
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

        t_in_incl = comp_time_in_incl(t_in_incl, incl_ind, isIn,
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
        arrival_time[isOut] = t # Correction TO DO - (xp[isOut] - Lx)/uxp

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

    return arrival_time, t_in_incl

#####
def transport_ds(grid, incl_ind, Npart, ux, uy, ds, isPeriodic=False):
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

    arrival_time = np.zeros(Npart)

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

        t_in_incl = comp_time_in_incl(t_in_incl, incl_ind, isIn,
                                      indx, indy, arrival_time)

        uxp = (ux[ix] + Ax[ix]*(xp[isIn] - x1[indx]))
        uyp = (uy[ix] + Ay[ix]*(yp[isIn] - y1[indy]))

        vp = np.sqrt(uxp*uxp + uyp*uyp)

        xp[isIn] = xp[isIn] + ds*uxp/vp
        yp[isIn] = yp[isIn] + ds*uyp/vp

        arrival_time[isIn] = arrival_time[isIn] + ds/vp

        #try:
        out = np.where(xp[isIn] - Lx >= 0.)
        arrival_time[out] = arrival_time[out] - (xp[out] - Lx)/uxp[out]

        #except:
            #ipdb.set_trace()

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

        t_in_incl = comp_time_in_incl(t_in_incl, incl_ind, isIn,
                                      indx, indy, arrival_time)
        isIn = np.where(xp < Lx)[0]

    

    print("Particles inside: %e" %( np.sum(isIn)))
    print("End of transport.")

    return  arrival_time, t_in_incl

####
def plotXY(x, y, fig=None, ax=None, lin=None, allowClose=False):

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
def plot2D(C, Nx, Ny, Lx, Ly, fig=None, ax=None, title=None, allowClose=False):

    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.gca()

    #Create X and Y meshgrid
    dx = Lx/Nx
    dy = Ly/Ny

    # xx, yy need to be +1 the shape of C because
    # pcolormesh needs the quadrilaterals.
    # Otherwise the last column is ignored.
    #see: matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.pcolor
    cny, cnx = C.shape

    if cnx > Nx:      # x face centered

        x1 = np.arange(-dx/2., Lx + dx, dx)
        y1 = np.arange(0., Ly + dy, dy)

    elif cny < Ny: # y face centered

        x1 = np.arange(0., Lx + dx, dx)
        y1 = np.arange(-dy/2., Ly + dy, dy)

    else:              #cell centered

        x1 = np.arange(0., Lx + dx, dx)
        y1 = np.arange(0., Ly + dy, dy)

    xx, yy = np.meshgrid(x1, y1)

    plt.sca(ax)
    plt.ion()

    mesh = ax.pcolormesh(xx, yy, C, cmap='coolwarm')

    #plt.axis('equal')
    #plt.axis('tight')
    plt.axis('scaled')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(mesh, cax=cax)

    if title is not None:
        plt.title(title)

    plt.show()

    if allowClose:
        input("Dale enter y cierro...")
        plt.close()
        fig = None
        ax = None

    return fig, ax
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

    fileending = filename.partition('.')[2]
    if fileending == 'dat':
        data = np.loadtxt(filename)
        Npart = len(data)
        arrival_time = data
        return  Npart, arrival_time # Npart, t_in_incl if imported from plk

    elif fileending == 'plk':
        with open(filename, 'rb') as ff:
            data = pickle.load(ff)
            Npart = data[0]
            t_in_incl = data[1]
            arrival_time = data[2]
            return  Npart, t_in_incl, arrival_time

################
def time_distributions(Npart, time_in_incl, writeit=False, fname=None):
    '''Returns the cummulative time each particle spent in all
       visited inclusions and the time particles spent in each inclusion.
    '''

    #dictionary with the total time each particle spent in any inclusion.
    tt = {}
    for ip in range(Npart):
        tt[ip] = 0.0

    #dictionary with the time particles spent in each inclusion.
    incl_times = {}

    i = 0
    tot_time_in_incl = total_time_in_incl(time_in_incl)
    for incl in tot_time_in_incl:
        incl_times[i] = np.array(list(incl.values()))

        tt = {key: tt.get(key, 0) + incl.get(key, 0)
              for key in set(tt) | set(incl)}

        i = i + 1

    particle_times = np.concatenate(np.vstack(list(tt.values())))

    if writeit:
    #Write time distribution in each particle.
        import pandas as pd

        df = pd.DataFrame.from_dict(incl_times, orient='index')
        df = df.transpose()
        df.columns = ((str(i + 1)  for i in list(incl_times.keys())))
        df.rename(columns={df.columns[0]:'#1'}, inplace=True)

        try:
            outfile = fname + '-time-incl.dat'
        except:
            outfile = 'time-incl.dat'

        df.to_csv(outfile, sep=' ', na_rep='?', index=False)

     #writes all particle times.
        try:
            outfile = fname + '-time-part.dat'
        except:
            outfile = 'time-part.dat'

        np.savetxt(outfile, particle_times, delimiter=' ')

    return incl_times, particle_times

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

    vals, edges, _ = plt.hist(data, bins=bins, normed=True)
    plt.xlabel('time')
    plt.ylabel('freq')
    plt.title(title)

    if savedata:
        filename = figname + '.dat'
        edges = (edges[:-1] + edges[1:])/2.0
        np.savetxt(filename, (edges, vals))

    if savefig:
        if figformat == 'pdf':
            print(figname)
            plt.savefig(figname + '.pdf', format='pdf')
        if figformat == 'tikz':
            from matplotlib2tikz import save as tikz_save
            tikz_save(figname + '.tex')

    if showfig:
        plt.show()
        plt.ion()
        input("Dale enter y cierro...")

    plt.close()
    return True
################
def postprocess(fname, writeit=True, showfig=False,
                savepartfig=True, saveinclfig=False, figformat='pdf',
                bins='auto'):
    """Post process from plk file. Computes the histograms for
       particles and inclusions. Saves data and/or figures.
    """

    Npart, t_in_incl = load_data(fname + '.plk')

    incl_times, particle_times = time_distributions(Npart, t_in_incl,
                                                    writeit=writeit,
                                                    fname=fname)

    #particle histogram
    if showfig or savepartfig:
        plot_hist(particle_times, title=fname + ' particles', bins=bins,
                  showfig=showfig, savefig=savepartfig, savedata=False,
                  figname=fname + '-part-hist', figformat=figformat)

    if showfig or saveinclfig:
        inclusions_histograms(incl_times, showfig=showfig, savefig=saveinclfig,
                              savedata=False, fname=fname,
                              figformat=figformat, bins=bins)
    return True
################
def postprocess_all(writeit=True, showfig=False,
                    savepartfig=True, saveinclfig=False, figformat='pdf',
                    bins='auto'):
    """ Post process al the cases in a folder."""

    import os as os

    files = os.listdir()
    for file in files:
        if file.endswith('plk'):
            fname  = os.path.splitext(file)[0]
            postprocess(fname, writeit=writeit, showfig=showfig,
                        savepartfig=savepartfig, saveinclfig=saveinclfig,
                        figformat=figformat, bins=bins)
################
def stream_function(grid, kperm, isPeriodic=False, plotPsi=False):
    '''Compute the stream function.
    The stream function is prescribed at the boundaries so that
    It is equivalent to prescribed flow on the left and right
    boundaries and no flow on top and bottom boundaries.'''

    isPeriodic = False #TO DO
    Lx, Ly, Nx, Ny = unpack_grid(grid)

    dx = Lx/(Nx-1)
    dy = Ly/(Ny-1)
    dx2 = dx*dx
    dy2 = dy*dy
    Np = Nx*Ny

    D1x, D2x, D1y, D2y = fd_mats(Nx, Ny, dx, dy)

    Y = np.log(kperm).reshape(Np, order='F')
    #(aa.T*bb).T
    Kmat = D1x.multiply(D1x*Y) + D1y.multiply(D1y*Y)
    #Kmat = (D1x.*(D1x.dot(Y))).T + (D1y.T*(D1y.dot(Y))).T
    #Kmat = bsxfun(@times, D1x*Y, D1x) + bsxfun(@times, D1z*Y, D1z)

    Amat = D2y + D2x - Kmat
    RHS = np.zeros(Np)
    #BC
    #
    #Top boundary
    Amat[0:Np:Ny,:] = 0.0
    idx = np.arange(0, Np, Ny)
    Amat[idx, idx] = 1.0
    RHS[0:Np:Ny] = Ly
    #
    #Bottom boundary
    Amat[Ny-1:Np:Ny,:] = 0.0
    idx = np.arange(Ny-1, Np, Ny)
    Amat[idx,idx] = 1.0
    RHS[Ny-1:Np:Ny] = 0.0
    #
    #Left boundary
    Amat[0:Ny,:] = 0.0
    idx = np.arange(0, Ny, 1)
    Amat[idx,idx] = 1.0
    RHS[0:Ny] = np.arange(0.,Ly+dy,dy)[::-1] #(Ly:-dy:0)
    #
    #Right boundary
    Amat[Np-Ny:Np, :] = 0.0
    idx = np.arange(Np-Ny, Np, 1)
    Amat[idx,idx] = 1.0;
    RHS[Np-Ny:Np] = np.arange(0.,Ly+dy,dy)[::-1] #(Ly:-dy:0)

    psi = lgsp.spsolve(Amat, RHS).reshape(Ny, Nx, order='F')
    if plotPsi:
        plot2D(psi, Nx, Ny, Lx, Ly, title='psi', allowClose=True)
    #
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
def comp_time_in_incl(t_in_incl, incl_ind, isIn, indx, indy, time):

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
    # total time of particles in inclusions
    num_incl = len(t_in_incl)
    incl = np.arange(0, num_incl, 1)
    for dic1, incl in zip(t_in_incl, incl):
        t_in_incl[incl] = {k:np.diff(v) for k,v in dic1.items()}

    return t_in_incl
################
def compute_cbtc(particle_times):
    cbtc_time = np.sort(particle_times)
    Npart = particle_times.shape[0]
    cbtc =  np.cumsum(np.ones(Npart))/Npart

    return cbtc_time, cbtc

################
def mobile_inmmobile_time(t_in_incl, arrival_times):
#TO DO
    return tmobile, tinmobile
