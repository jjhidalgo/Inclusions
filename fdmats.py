
# D1x, D2x, D1y, D2y = fd_mats(Nx, Ny, dx, dy, isPeriodic=isPeriodic)


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
