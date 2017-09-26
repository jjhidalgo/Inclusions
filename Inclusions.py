import pdb
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lgsp
################
def permeability(nry, Lx, Nx, Ny, Kfactor, pack='sqr', plotit=False):

    from RecPore2D import RegPore2D as rg

    Ly = 1.
    
    nrx = np.int(Lx*nry)
    r = 1./np.sqrt(2.*np.pi*np.float(nry*nry))
    d = (1.- 2.*nry*r)/(nry + 1.)

    a = rg(nx=nrx, ny=nry, radius=r, packing=pack)
    a.throat = d
    a.bounding_box = ([0.0, 0.0, 0.5], [Lx, Ly, 1.0])
    a.xoffset = 0.

    #circles' centers
    delta = Lx - nrx*(d + 2.*r)

    circles = a.circles

    y0 = circles[:]['y']
    x0 = circles[:]['x'] + (delta-d)/2.
    circles[:]['x'] = x0
    a.circles = circles
    #a.write_mesh(fname='a.geo', meshtype='gmsh')

    kperm = np.ones([Ny, Nx])
    dx = Lx/Nx
    dy = Ly/Ny
    x1 = np.arange(dx/2., Lx, dx) #cell centers' coordinates
    y1 = np.arange(dy/2., Ly, dy)
    xx, yy = np.meshgrid(x1, y1)


    for circ in circles:
        x0 = circ['x']
        y0 = circ['y']
        r = circ['r']
        kperm[((xx - x0)**2. + (yy - y0)**2.) < r**2.] = Kfactor

    #kperm = np.sqrt(xx*xx + yy*yy) 
    if plotit:
        plot2D(kperm, Nx, Ny, Lx, Ly)
        plt.title('kperm')
        raw_input("Dale enter y cierro...")
        plt.close()
    #kperm = np.ones(kperm.shape)
    #kperm[yy>0.2] = 0.1
    #kperm[yy>0.5] = 0.5
    
    return kperm

################
def flow(Nx, Ny, Lx, Ly ,kperm, bcc, plotHead=False):

    dx = Lx/Nx
    dy = Ly/Ny
    dx2 = dx*dx
    dy2 = dy*dy
    mu = 1./kperm
    Np = Nx*Ny

    # Transmisibility matrices.
    Tx = np.zeros((Ny, Nx + 1))
    Tx[:, 1:Nx] = (2.*dy)/(mu[:, 0:Nx-1] + mu[:, 1:Nx+1])

    Ty = np.zeros([Ny + 1, Nx])
    Ty[1:Ny, :] = (2.*dx)/(mu[0:Ny-1, :] + mu[1:Ny+1, :])

    Tx1 = Tx[:, 0:Nx].reshape(Np, order='F')
    Tx2 = Tx[:, 1:Nx+1].reshape(Np, order='F')

    Ty1 = Ty[0:Ny, :].reshape(Np, order='F')
    Ty2 = Ty[1:Ny+1, :].reshape(Np, order='F')

    Ty11 = Ty1
    Ty22 = Ty2

    TxDirich = np.zeros(Ny)
    TxDirich = (2.*dy)*(1./mu[:, Nx-1])
    if bcc:
        TxDirichL = (2.*dy)*(1./mu[:, 0])

    #Assemble system of equations
    Dp = np.zeros(Np)
    Dp = Tx1/dx2 + Ty1/dy2 + Tx2/dx2 + Ty2/dy2
    Dp[Np-Ny:Np] = Ty1[Np-Ny:Np]/dy2 + Tx1[Np-Ny:Np]/dx2 + \
                     Ty2[Np-Ny:Np]/dx2 + TxDirich/dx2

    if bcc:
        Dp[0:Ny] = Ty1[0:Ny]/dy2 + Tx2[0:Ny]/dx2 + \
                     Ty2[0:Ny]/dx2 + TxDirichL/dx2

    Am = sp.spdiags([-Tx2/dx2, -Ty22/dy2, Dp,
                     -Ty11/dy2, -Tx1/dx2], [-Ny, -1, 0, 1, Ny], Np, Np,
                    format='csr')


    #RHS - Boundary conditions
    u0 = 1.
    hL = 1.
    hR = 0.
    S = np.zeros(Np)
    S[Np-Ny:Np+1] = TxDirich*(hR/dx/dx) # Dirichlet X=1;
    if bcc:
        S[0:Ny] = TxDirichL*(hL/dx/dx) # Dirichlet X=0;
    else:
        S[0:Ny] = u0/dx #Neuman BC x=0;

    head = lgsp.spsolve(Am, S).reshape(Ny, Nx, order='F')

    #Compute velocities

    ux = np.zeros([Ny, Nx+1])
    uy = np.zeros([Ny+1, Nx])

    if bcc:
        ux[:, 0] = -TxDirichL*(head[:, 0] - hL)/dx
    else:
        ux[:, 0] = u0

    ux[:, 1:Nx] = -Tx[:, 1:Nx]*(head[:, 1:Nx+1] - head[:, 0:Nx-1])/dx
    ux[:, Nx] = -TxDirich*(hR - head[:, Nx-1])/dx

    uy[1:Ny, :] = -Ty[1:Ny, :]*(head[1:Ny+1, :] - head[0:Ny-1, :])/dy
    
    
    if plotHead:
        plot2D(head, Nx, Ny, Lx, Ly)
        plt.title('head')
        raw_input("Dale enter y cierro...")
        plt.close()
        plot2D(ux/dy, Nx, Ny, Lx, Ly)
        plt.title('ux')        
        print(np.max(ux/dy))
        print(np.min(ux/dy))              
        print(np.mean(ux/dy))
        print(np.var(ux/dy))
        raw_input("Dale enter y cierro...")
        plt.close()
        plot2D(uy/dx, Nx, Ny, Lx, Ly)
        plt.title('uy')        
        raw_input("Dale enter y cierro...")
        plt.close()

    return ux/dy, uy/dx

#####
def transport(Npart, ux, uy, Nx, Ny, Lx, Ly, tmax, dt, plotit = False, kperm=None):

    if plotit:
        figt, axt = plot2D(uy, Nx, Ny,Lx, Ly)

    t = 0.
    xp = np.zeros(Npart)
    yp = np.random.rand(Npart)

    dx = np.float(Lx/Nx)
    dy = np.float(Ly/Ny)

    Ax = ((ux[:, 1:Nx + 1] - ux[:, 0:Nx])/dx).flatten(order='F')
    Ay = ((uy[1:Ny + 1, :] - uy[0:Ny, :])/dy).flatten(order='F')

    ux = ux[:, 0:Nx].flatten(order='F')
    uy = uy[0:Ny, :].flatten(order='F')

    x1 = np.arange(0., Lx + dx, dx) #faces' coordinates
    y1 = np.arange(0., Ly + dy, dy)

    time = np.zeros(1)
    cbtc = np.zeros(1)

    i = 0
    
    lint = None
    isIn = True
    
    while t <= tmax and np.sum(isIn) > 0:
        #remove partcles
        isIn = np.where(xp < Lx)
        t = t + dt
        i = i + 1
        #indx = np.minimum(np.int_(xp[isIn]/dx), Nx-1)#.astype(int)
        #indy = np.minimum(np.int_(yp[isIn]/dy), Ny-1)#.astype(int)
        indx = np.int_(xp[isIn]/dx)
        indy = np.int_(yp[isIn]/dy)
        ix = (indy + indx*Ny)

        xp[isIn] = xp[isIn] + (ux[ix] + Ax[ix]*(xp[isIn] - x1[indx]))*dt
        yp[isIn] = yp[isIn] + (uy[ix] + Ay[ix]*(yp[isIn] - y1[indy]))*dt

        #print ["{0:0.19f}".format(i) for i in yp]

        xp[xp < 0.] = 0.
        
        #boudary reflection            
        yp[yp < 0.] = -yp[yp < 0.]
        yp[yp > Ly] = 2.*Ly - yp[yp > Ly]

        if i%10 == 0:
            cbtc = np.append(cbtc, np.sum(xp > Lx)/np.float(Npart))
            time = np.append(time, t)
            print(t)
            if plotit:                    
                figt, axt, lint = plotXY(xp[isIn], yp[isIn], figt, axt, lint)
                
                axt.set_aspect('equal')
                axt.set_xlim([0., Lx])
                axt.set_ylim([0., Ly])
                plt.title(t)
                
    return cbtc, time
#####
def plotXY(x, y, fig=None, ax=None, lin=None):

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

    return fig, ax, lin
####
def plot2D(C, Nx, Ny,Lx, Ly, fig=None, ax=None):

    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.gca()

#Create X and Y meshgrid
    dx = Lx/Nx
    dy = Ly/Ny

    # xx, yy need to be +1 the shape of C because
    #pcolormesh needs the quadrilaterals.
    #Otherwise last column is ignored.
    #see: matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.pcolor
    cny, cnx = C.shape

    if cnx > cny:      #x face centered

        x1 = np.arange(-dx/2., Lx + dx, dx) 
        y1 = np.arange(0., Ly + dy, dy)
        
    elif cnx < cny: #y face centered

        x1 = np.arange(0., Lx + dx, dx) 
        y1 = np.arange(-dy/2., Ly + dy, dy)
        
    else:              #cell centered

        x1 = np.arange(0., Lx+dx, dx) 
        y1 = np.arange(0., Ly+dy, dy)
        
    xx, yy = np.meshgrid(x1, y1)


    plt.sca(ax)
    plt.ion()

    mesh = ax.pcolormesh(xx, yy, C, cmap='coolwarm')

    #plt.axis('equal')
    #plt.axis('tight')
    plt.axis('scaled')
    plt.colorbar(mesh, ax=ax)
    plt.show()

    return fig, ax

#####
# Ly = 1.
# Lx = 1.
# Ny = 5
# Nx = np.int(Ny*Lx)
# Kfactor = 0.1
# nry = 5
# plotit = False
# kperm = permeability(nry, Lx, Nx, Ny, Kfactor, plotit)

# #kperm = np.ones(kperm.shape)
# bcc = True
# ux, uy = flow(Nx, Ny, Lx, Ly, kperm, bcc)

# tmax = 1e2
# dt = 1e-3

# Npart = np.int(1e5)
# cbtc, time = transport(Npart, ux, uy, Nx, Ny, Lx, Ly, tmax, dt)

# np.savetxt('cbtc.dat', np.matrix([time, cbtc]).transpose())
# figc = None
# axc = None
# linc = None
# figc, axc, linc = plotXY(time, 1-cbtc, fig=figc, ax=axc, lin=linc)
# #pdb.set_trace()
# raw_input("Press enter to continue...")
