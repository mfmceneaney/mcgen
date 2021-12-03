# Graph results from file
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

@njit
def np_std(data):
    return np.std(data)/np.sqrt(np.shape(data)[0])

@njit
def block_indices(data,step=100,nsigma=2,trail=100):
    
    """
    arr    - 1D input np.array
    step   - step size for checking running average
    nsigma - # of stddevs away the new mean has to be to initiate new block
    trail  - array size for trailing mean
    """
    
    indices = [0] # Append separating indices for partitioning here, always include first index
    for k in range(step,len(data)-1,step):
        if nsigma*np_std(data[max(indices[-1],k-trail):k])<abs(np.mean(data[k:k+step])-np.mean(data[max(indices[-1],k-trail):k])):
            indices.append(k)
                
    #Make sure you go all the way to the end
    if indices[-1]!=len(data)-1: indices.append(len(data))
    
    return indices

@njit
def block(data,step=100,nsigma=2,trail=100):
    
    """
    arr    - 1D input np.array
    step   - step size for checking running average
    nsigma - # of stddevs away the new mean has to be to initiate new block
    trail  - array size for trailing mean
    """

    indices = [0] # Append separating indices for partitioning here, always include first index
    for k in range(step,len(data)-1,step):
        if nsigma*np_std(data[max(indices[-1],k-trail):k])<abs(np.mean(data[k:k+step])-np.mean(data[max(indices[-1],k-trail):k])):
            indices.append(k)
                
    #Make sure you go all the way to the end
    if indices[-1]!=len(data)-1: indices.append(len(data))
    
    # Get blocked stddev
    sigma = 0
    for k in range(len(indices)-1): #IMPORTANT: Extra factor of sqrt(1/N) for sigma but squared
        sigma += np.square(np_std(data[indices[k]:indices[k+1]])) 
    sigma = np.sqrt(sigma)
    sigma /= len(indices)-1 #IMPORTANT: Divide by N AFTER
    
    return sigma

@njit
def statistics(data,idx,i1=0,i2=10000,step=100,nsigma=2,trail=100): # Return mean and stddev/sqrt(N)
    if step==-1:return [np.mean(data[i1:i2,idx]), np_std(data[i1:i2,idx])]
    return [np.mean(data[i1:i2,idx]), block(data[i1:i2,idx],step=step,nsigma=nsigma,trail=trail)]

@njit
def statistics_indices(data,idx,i1=0,i2=10000,step=100,nsigma=2,trail=100): # Return mean and stddev
    
    #No Blocking option
    if step==-1: return [[np.mean(data[i1:i2,idx])],[np.std(data[i1:i2,idx])/np.sqrt(len(data[i1:i2,idx]))]]
    
    indices = block_indices(data[i1:i2,idx],step=step,nsigma=nsigma,trail=trail)
    means  = []
    sigmas = []
    for k in range(len(indices)-1):
        m = np.mean(data[i1:i2,idx][indices[k]:indices[k+1]])
        s = np_std(data[i1:i2,idx][indices[k]:indices[k+1]])
        for _ in range(indices[k],indices[k+1]):
            means.append(m)
            sigmas.append(s)
    means = np.array(means)
    sigmas = np.array(sigmas)
    
    return [means,sigmas]

def plot_blocks(data,name,idx,i1=0,i2=10000,step=100,nsigma=2,trail=100,figsize=(16,10)):

    if step==-1: return
    means, sigmas = statistics_indices(data,idx,i1,i2,step=step,nsigma=nsigma,trail=trail)
    highs = np.add(means,sigmas)
    lows  = np.add(means,-1*sigmas)
    x     = np.array([i for i in range(len(means))])

    plt.figure(figsize=figsize)
    plt.xlabel("iteration")
    plt.ylabel(name)
    plt.plot(x,means,marker='o',linewidth=0)
    plt.plot(x,highs,marker='o',linewidth=0,color='r')
    plt.plot(x,lows,marker='o',linewidth=0,color='r')
    plt.hist(x,weights=data[i1:i2,idx],bins=100,density=True)

    # Set axes limits
    ymin, ymax = 0, np.max(highs)+0.1
    ax = plt.gca()
    if name in ["N"]: ax.set_ylim([ymin, ymax])
    
    plt.show()

# Part 2: Plot MC Results and Fit
def plot_fit_obs(x, y, yerr=None, xlabel="$\epsilon$", ylabel="$<n>_\epsilon$", label="MonteCarlo Results", fmt="o", capsize=5, figsize=(8,5)):
    
    """
    Arguments:
        x - x axis points against which to plot operator MC ensemble average
        y - y axis points corresponding to operator MC ensemble averages
    """
    
    # Fit data
    def func(x0, f0, f1, f2, f3):
        return f0*np.ones(np.shape(x0))+f1*np.power(x0,1)+f2*np.power(x0,2)+f3*np.power(x0,3)
    popt, pcov = curve_fit(func, x, y)

    # Set font sizes
    plt.rc('axes', labelsize=18)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('legend', fontsize=12)
    
    # Plot data and fit as a function of time step size \epsilon
    plt.figure(figsize=figsize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.errorbar(x, y, yerr=yerr, fmt=fmt, capsize=capsize, label=label)
    plt.plot(x, func(x, *popt), 'r-',
             label='fit: $f_0$=%5.4f, $f_1$=%5.4f, $f_2$=%5.4f, $f_3$=%5.4f' % tuple(popt))
    plt.legend()
    plt.show()

    # Part 2: Plot MC Results and Fit
def plot_obs(x, y, yerr=None, xlabel="$\epsilon$", ylabel="$<n>_\epsilon$", label="MonteCarlo Results", fmt="o", capsize=5,figsize=(8,5)):
    
    """
    Arguments:
        x - x axis points against which to plot operator MC ensemble average
        y - y axis points corresponding to operator MC ensemble averages
    """

    # Set font sizes
    plt.rc('axes', labelsize=18)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('legend', fontsize=12)
    
    # Plot data and fit as a function of time step size \epsilon
    plt.figure(figsize=figsize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.errorbar(x, y, yerr=yerr, fmt=fmt, capsize=capsize, label=label)
    plt.legend()
    plt.show()

# Part 2: Plot MC Results and Fit
def plot_obs_together(x, ys, yerrs=None, xlabel="$\epsilon$", ylabel="$<n>_\epsilon$", labels=["MonteCarlo Results"], fmt="o", capsize=5,figsize=(8,5)):
    
    """
    Arguments:
        x - x axis points against which to plot operator MC ensemble average
        y - y axis points corresponding to operator MC ensemble averages
    """

    # Set font sizes
    plt.rc('axes', labelsize=18)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('legend', fontsize=12)
    
    # Plot data and fit as a function of time step size \epsilon
    plt.figure(figsize=figsize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for idx in range(len(ys)):
        plt.errorbar(x, ys[idx], yerr=(yerrs[idx] if yerrs!=None else None), fmt=fmt, capsize=capsize, label=labels[idx])
    plt.legend()
    plt.show()
