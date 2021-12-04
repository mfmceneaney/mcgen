import numpy as np
from numpy.linalg import eig
from scipy.linalg import expm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Functions

def basis(N,d,nStates=2):
    
    """
    Arguments:
        N       - number of lattice sites along one spatial axis
        d       - dimension of physical space e.g., 3D, 2D, 1D
        nStates - # of states at each site (2 for hardcore bosons)
    Returns:
        basis   - np.array of all possible particle configuration vectors
    """
    
    dim    = N**d
    helper = [[k] for k in range(nStates)]
    for _ in range(dim-1):
        helper = [ el+el2 for el2 in [[k] for k in range(nStates)] for el in helper ]
    basis = np.array(helper)
    
    return basis

def coords(N,d):
    
    """
    Arguments:
        N     - number of lattice sites along one spatial axis
        d     - dimension of physical space e.g., 3D, 2D, 1D
    Returns:
        coords - np.array of all site coordinates (assuming lattice spacing of 1)
    Note:
        This is essentially the same function as basis() above but d can be anything.
    """
    
    helper = [[k] for k in range(N)]
    for _ in range(d-1):
        helper = [ el+el2 for el2 in [[k] for k in range(N)] for el in helper ]
    coords = np.array(helper)
    
    return coords

def units(d):
    
    """
    Arguments:
        d     - dimension of physical space e.g., 3D, 2D, 1D
    Returns:
        units - np.array of unit vectors in d-dimensional space
    """
    
    return np.array([[1 if i==idx else 0 for i in range(d)] for idx in range(d)])

def bc(v,N):
    
    """
    Arguments:
        v   - input vector of same dimension as the physical space e.g., 3D, 2D, 1D
        N   - # of particles along a given lattice axis
    Returns:
        res - resulting array from action of the periodic boundary conditions
    """

    res = v.copy()
    res[res > N-1] = 0
    res[res < 0]   = N-1
    return res

def flip(v,i,j):
    
    """
    Arguments:
        v - basis vector on which to flip particle at ith position to jth position
        i - ith index to flip from
        j - jth index to flip to
    Returns:
        u - flipped vector
    """

    u = v.copy()
    u[i] = 0
    if u[j] != 1:
        u[j] = 1
        return u
    else:
        return -np.ones(np.shape(u)) # Impose hardcore constraint here

def flip3D(v,i,sign,N,coords,units):
    
    """
    Arguments:
        v      - basis vector on which to flip particle at ith position to jth position
        i      - ith index to flip from
        sign   - flip in positive or negative direction 
        N      - # of particles along a given lattice axis
        coords - spatial coordinates of lattice sites
        units  - spatial unit vectors
    Returns:
        l      - np.array of 3D flipped vectors
    """
    
    # Loop lattice sites and unit vectors to find neighbors
    if v[i]==0: return np.array([-np.ones(np.shape(v))])
    l = np.array([
        flip(v,i,idx) if np.array_equal(coords[idx],bc(coords[i]+sign*unit,N))
         else -np.ones(np.shape(v))
        for unit in units
        for idx in range(len(coords))
    ])
                    
    return l
    
def cross(v,N,coords,units):
    
    """
    Arguments:
        v      - basis vector for which to compute cross terms of Hamiltonian
        N      - # of particles along a given lattice axis
        coords - spatial coordinates of lattices sites
        units  - spatial unit vectors
    Returns:
        l      - np.array of resulting basis vectors
    """

    l = np.array([
        np.concatenate((flip3D(v,i,1,N,coords,units),flip3D(v,i,-1,N,coords,units)))
        for i in range(len(v))
    ])
    l = np.concatenate(l,axis=0)
    
    return l

def hamiltonian(t,mu,N,d=3,nStates=2):
    
    """
    Arguments:
        t       - t parameter of Hamiltonian
        mu      - mu parameter of Hamiltonian
        N       - # of sites or particles
        d       - # of spatial dimensions for basis
        nStates - # of states at each site (filled or empty in this case)
    Returns:
         H      - np.array Hamiltonian of dimension (nStates**(N**d) x nStates**(N**d))
    """

    # Create and fill Hamiltonian
    b   = basis(N,d,nStates)
    c   = coords(N,d)
    u   = units(d)
    dim = nStates**(N**d)
    H   = np.array([
        [(6*t-mu)*np.sum(b[i]) if i==j else 0
         for i in range(dim)]
        for j in range(dim)
    ])
    if t!=0: # This step takes a long time
        H   -= t*np.array([
            [ 
                np.sum([1 if np.array_equal(b[i],el) else 0 for el in cross(b[j],N,c,u)])
                for i in range(dim)
            ] 
            for j in range(dim)
        ])
    
    return H

def num(basis=basis(2,3,2)):

    """
    Arguments:
        basis - basis of particle number at each lattice site
    Returns:
        num   - diagonal dim(basis) x dim(basis) matrix with number operator eigen values in basis
    """
    
    num = np.array([
        [np.sum(basis[i]) if i==j else 0 
         for i in range(len(basis))]
        for j in range(len(basis))
    ])
    
    return num


def ensAve(h,o,b=12):
    
    """
    Arguments:
        h  - Hamiltonian matrix
        o  - operator matrix (in same basis as h)
        b  - beta parameter for ensemble average
    Returns:
        ave - expectation value (ensemble average) of operator
    """
    
    # Diagonalize Hamiltonian
    r  = eig(h)
    hD = np.diag(r[0])
    u  = r[1].T #IMPORTANT: This must be transpose!
    
    # Calculate ensemble average
    z   = np.trace(expm(-b*h))
    ave = 1/z*np.trace(
                        np.matmul(o,expm(-b*h)
                    ))
    return ave

def plotEnsAveVsMu(o,t,muRange,muStep,b,N=2,d=3,ylabel="<n>",figsize=(8,5)):
    
    """
    Arguments:
        o       - operator for which to take ensemble average
        t       - t parameter in definition of Hamiltonian
        muRange - range in mu over which to plot <o>
        muStep  - step size for muRange
        b       - beta parameter for ensemble average
        N       - # of sites or particles
        d       - # of spatial dimensions for basis
    """
    
    # Construct Hamiltonian and number operator here to save time
    h0  = hamiltonian(1,0,N,d)
    h1  = hamiltonian(0,1,N,d)
    
    # Get data points
    nSteps = int((muRange[1]-muRange[0])/muStep) + 1
    xData  = np.linspace(muRange[0],muRange[1],nSteps)
    yData  = [ensAve(h0*t+h1*(muRange[0]+n*muStep),o,b) for n in range(nSteps)]
    
    plt.rc('axes', labelsize=18) #fontsize of the y tick labels
    plt.rc('xtick', labelsize=12) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=12) #fontsize of the y tick labels
    
    # Plot data points
    plt.figure(figsize=figsize)
    plt.plot(xData,yData,marker="o",linewidth=1,linestyle="dashed")
    plt.xlabel("$\mu$")
    plt.ylabel(ylabel)
    plt.show()
