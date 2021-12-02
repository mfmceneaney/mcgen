#--------------------------------------------------#
# PHYS765 Project 1 Part 1
# Script to exactly diagonalize 
#--------------------------------------------------#

import numpy as np
from exactlib import *

# Suppress warnings
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# Part 1: Set parameters
t   = 1
mu  = 1.4
b   = 12
N   = 2 # Dimension along single axis
d   = 3 # Number of spatial dimensions

# Initialize Hamiltonian
h = hamiltonian(t,mu,N,d)
h0 = h - hamiltonian(0,mu,N,d) # Saves time to compute like this

# Get ensemble averages
nAve = ensAve(h,num(),b)
eAve = ensAve(h,t*h0,b)
print("<n> = ",nAve)
print("<e> = ",eAve)

# Part 2: Set parameters
t       = 1
muRange = (0,5)
muStep  = 0.1
b       = 12
plotEnsAveVsMu(num(),t,muRange,muStep,b)
