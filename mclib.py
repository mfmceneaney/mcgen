#-------------------------------------------------------------------#
# PHYS 765 Project Fall 2021:                                       #
# Monte Carlo generation for Hard Core Bosons on a periodic lattice #
#                                                                   #
#                                                                   #
#-------------------------------------------------------------------#

import numpy as np
from numba import jit, njit, types
from numba.typed import Dict
from copy import deepcopy
from tqdm import tqdm

# Classes

class Lattice:
    
    def __init__(self,N=[2,2,2]):
        """
        Arguments:
            N - array containing # of lattice sites along each spatial axis (z,y,x)
        """
        self.N = N
        self.d = len(N)
        self.directions = [sign*d for sign in[1,-1] for d in range(1,len(N)+1)]
        self.sites      = np.array([i for i in range(np.product(N))])
        self.lattice    = np.copy(np.reshape(self.sites,N))
        #NOTE: Sites are numbered in x then y then z.
        #NOTE: However, indexing is, e.g., [z,y,x] in 3D.
        #NOTE: So unit vectors are also reversed, e.g. unit z = [1,0,0]
        self.neighbors  = {
            s:{ 
                int(d):self.convert(self.bc(np.squeeze(np.where(self.lattice==s)) + 
                          np.array([np.sign(d) if k==abs(d)-1 else 0 for k in range(len(N))],dtype=np.int)
                                    ))
                for d in self.directions }
            for s in self.sites}
        
    def bc(self,v):
        """
        Arguments:
            v - vector on which to apply periodic boundary conditions
        """
        res = v.copy()
        res[np.greater(res,self.N-np.ones_like(self.N))] = 0
        res[np.less(res,np.zeros_like(self.N))] = self.N[np.where(res<0)[0][0]]-1 if np.size(np.where(res<0)[0])!=0 else []  #NOTE: Only works because we move one by one
        return np.array(res,dtype=np.int)
    
    def convert(self,v):
        """
        Arguments:
            v - vector to be converted back to integer notation
        """
        res = v.copy()
        res = np.sum([res[-(k+1)]*np.product(self.N[0:k]) for k in range(len(self.N))],dtype=np.int)
        return res
    
    def getNeighbor(self,s,d):
        """
        Arguments:
            s - integer site #
            d - direction of neighbor (0->Self, 1->z, 2->y, 3->z)
        """
        if d==0: return s
        else: return self.neighbors[s][d]
    
    def getWN(self,s,d):
        """
        Note: d gets reset to an int to so that you don't get type errors for indices.
              Also, basically this checks if you are on the edge of the lattice going in the specified direction.
        """
        if d>0 and np.squeeze(np.where(self.lattice==s))[int(abs(d)-1)] == self.N[int(abs(d)-1)]-1: return 1
        elif d<0 and np.squeeze(np.where(self.lattice==s))[int(abs(d)-1)] == 0: return -1
        else: return 0

#---------- Jit Wrapped Functions for Configuration Class ----------#
@njit
def getTHops_(config):
    mysum = 0
    for timeslice in config:
        for site in range(config.shape[1]):
            if timeslice[site][1]==0: mysum += 1
    return mysum

@njit
def getSHops_(config,fill):
    mysum = 0
    for timeslice in config:
        for site in range(config.shape[1]):
            if (timeslice[site][1]!=0 and timeslice[site][1]!=fill): mysum += 1
    return mysum

@njit
def getW_(config,lattice,N,fill,direction=None):
    mysum = 0
    for timeslice in config:
        for site in range(config.shape[1]):
            if timeslice[site,1] == fill or (direction!=None and abs(timeslice[site,1])!=direction): continue
            d = timeslice[site,1]
            if d>0 and np.where(lattice==site)[int(abs(d)-1)][0] == N[int(abs(d)-1)]-1: mysum += 1
            elif d<0 and np.where(lattice==site)[int(abs(d)-1)][0] == 0: mysum += -1
                
    return mysum

@njit
def getW2_(config,lattice,N,fill,direction=None):
    mysum = 0
    for timeslice in config:
        for site in range(config.shape[1]):
            if timeslice[site,1] == fill or (direction!=None and abs(timeslice[site,1])!=direction): continue
            d = timeslice[site,1]
            if d>0 and np.where(lattice==site)[int(abs(d)-1)][0] == N[int(abs(d)-1)]-1: mysum += 1
            elif d<0 and np.where(lattice==site)[int(abs(d)-1)][0] == 0: mysum += -1
                
    return np.square(mysum)#DEBUGGING, I think this might be the correct way...

@njit
def checkSite_(config,t,s,fill):
    """
    Return True if site is empty
    NOTE: t, s are converted to int type if not already
    """
    return config[int(t),int(s),1] == fill

@njit
def setSite_(config,t,s,i,d):
    """
    t - time step
    s - site #
    d - direction #
    i - world line index
    NOTE: t, s should be converted to int type if not already
    """
    config[int(t),int(s)] = [i,d]
    
@njit
def delSite_(config,t,s,fill):
    """
    NOTE: t, s should be converted to int type if not already
    """
    config[int(t),int(s)] = [fill,fill]
    
@njit
def getSite_(config,t,s):
    """
    NOTE: t, s should be converted to int type if not already
    """
    return config[t,s]

@njit
def getPrevious_(config,t,s):
    """
    Note: t, s should be converted to int type if not already
    """
    return config[t,s,0]

#-------------------------------------------------------------------#
class Configuration:
    
    def __init__(self,k,N=[2,2,2],fill=-9999):
        """
        Arguments:
            k    - # of timesteps (should be an integer, will get converted to integer type if not already)
            N    - array containing # of lattice sites along each spatial lattice
            fill - fill value for empty sites/directions
        """
        
        self.k      = int(k)
        self.N      = N
        self.d      = len(N) # Spatial dimension
        self.fill   = fill
        self.config = fill * np.ones((int(k),np.product(N),2)) # Store wl id # and direction at each space-time point
    
    def checkSite(self,t,s):
        """
        Return True if site is empty
        NOTE: t, s are converted to int type if not already
        """
        return checkSite_(self.config,t,s,self.fill)#int(t),int(s)#DEBUGGING SPEEDUPS
    
    def setSite(self,t,s,i,d):
        """
        t - time step
        s - site #
        d - direction #
        i - world line index
        NOTE: t, s are converted to int type if not already
        """
        setSite_(self.config,t,s,i,d)#DEBUGGING SPEEDUPS int(t),int(s)
        
    def delSite(self,t,s):
        """
        NOTE: t, s are converted to int type if not already
        """
        delSite_(self.config,t,s,self.fill)#DEBUGGING SPEEDUPS int(t),int(s)

    def getSite(self,t,s):
        """
        NOTE: t, s are converted to int type if not already
        """
        return self.config[int(t),int(s)]
        
    def getPrevious(self,t,s):
        """
        Note: t, s are converted to int type if not already
        """
        return self.config[int(t),int(s),0]
        
    def getN(self):
        return len(self.config[0,:,0][self.config[0,:,0]!=self.fill]) #NOTE: dropped np.unique since based on Nov. 10th office hrs don't need to label world lines differently
    
    def getTHops(self):
        return getTHops_(self.config)
    
    def getSHops(self):
        return getSHops_(self.config,self.fill)
    
    def getE(self,beta,t,mu):#NOTE: Approximation from lecture 10
        return -1/beta * self.getSHops() + 2*(t * self.d - mu)*self.getN()
    
    def getET(self,beta,t):#NOTE: Approximation from lecture 10
        return -1/beta * self.getSHops() + 2*(t * self.d)*self.getN()
    
    def getW(self,lattice,d=None):
        return getW_(self.config,lattice,np.array(self.N),self.fill,direction=d)
    
    def getW2(self,lattice,d=None):
        return getW2_(self.config,lattice,np.array(self.N),self.fill,direction=d)
    
    def getObs(self,lattice,beta,t,mu,keys):
        return np.array([
            self.getN()                     if key=="N"  else
            self.getE(beta,t,mu)            if key=="E"  else
            self.getET(beta,t)              if key=="ET" else
            self.getW(lattice.lattice,d=1)  if key=="W"  else # By Default just look in 1D here along z
            self.getW2(lattice.lattice,d=1) if key=="W2" else # By Default just look in 1D here along z
            self.getSHops()                 if key=="SHops" else
            self.getTHops()                 if key=="THops" else
            self.fill for key in keys
        ])
    
    def clone(self):
        return deepcopy(self)

    def sanityCheck(self,lattice): #DEBUGGING...
        """
        Checks if all sites and directions match sites and directions in next time slice (checks world line continuity basically)
        Note: Not completely sure about this still.
        """
        return np.all([
            True if 
                  self.config[int(timeslice%self.k)][int(site)][0]==self.fill
                  else self.config[int((timeslice-1)%self.k)][int(self.config[int(timeslice%self.k)][int(site)][0])][0]!=self.fill
            for site in range(self.config.shape[1])
            for timeslice in range(self.config.shape[0])
        ])
        

#---------- Jit wrapped functions for Worm class ----------#
@njit
def np_rand_random():
    return np.random.random()

@njit
def np_rand_randint(l,h):
    return np.random.randint(l,h)

@njit
def rand_direction(directions,l,h):
    return directions[np.random.randint(l,h)]

@njit
def apply_(t1,mu,e,k,dim,t,s,i,d,a,config,dmap,directions,fill):#config

        # Pulled these out to help speed up the loop
        forward_thresh  = np.exp(mu*e)
        backward_thresh = np.exp(-mu*e)
        remain_thresh   = (1-2*dim*t1*e)

        while True:

            """
            t - time slice #
            s - site #
            i - world line index
            d - direction # (for next step, set to 0 if staying on same site)
            a - arrow (?)
            """
            
            # NO PARTICLE AT SITE
            if config[int(t),int(s),1]==fill:

                p = np.random.random() # get random float in [0,1)
                # check arrow direction
                if a == 1: 

                    # PROPOSE FORWARD
                    if p < forward_thresh: # ACCEPTED #np.exp(self.mu*self.e)         
                        p = np.random.random()

                        if  p < remain_thresh: # REMAIN AT SAME SITE #(1-2*self.d*self.t*self.e)
                            config[int(t),int(s)]=[i,0] 
                            t, i = (t+1)%k, s
                            continue # Re-enter the algorithm at the top
                        
                        else: # MOVE TO DIFFERENT SITE
                            config[int(t),int(s)]=[i,d] #NOTE: This had the same typo as above: changed (t,s,d,i) to (t,s,i,d)
                            i = s
                            s = dmap[int(s)][int(d)]
                            d = directions[np.random.randint(0,2*dim)]
                            t = (t+1)%k
                            continue # Re-enter the algorithm at the top

                    else: # PROPOSE FORWARD REJECTED # just flip direction/arrow to back and not move in this step
                        a = -1
                        continue # Re-enter the algorithm at the top
                elif a == -1:
                    break

            else: # ALREADY PARTICLE AT SITE
                i1, d1 = config[int(t),int(s)]
                if a == 1: # if we got here from moving forward # basing this on slide 46, where he just flips arrow and then seems to let backwards proposal take care of it                     
                    a=-1
                    config[int(t),int(s)]=[i,d1] # Connect world lines#NOTE: Reset the "previous" entry at the colliding site so that the world line is continuous
                    if (i1==fill or config[int((t-1)%k),int(i1),1]==fill):#NOTE: DEBUGGING have an or for if you search previous time slice at requested previous site and find no particle
                        break

                     # Begin again at the previous site from the world line you collided with
                    s1 = i1 #Store previous site index
                    i2 = config[int((t-1)%k),int(s1),0] #Grab previous of previous site
                    if (i2==fill or config[int((t-2)%k),int(i2),1]==fill): #NOTE: Need this break here because you have to eat up the world line to the next time slice and if it's the end just end, should be taken care of below but apparently not
                        config[int((t-1)%k),int(s1)] = [fill,fill]
                        break

                    t, s, i = (t-1)%k, s1, i2
                    continue # Re-enter the algorithm at the top

                elif a == -1: # if we got here from moving backwards, then I think we just propose backward again
                    p = np.random.random()

                    # PROPOSE BACKWARDS
                    if p < backward_thresh: #np.exp(-1*self.mu*self.e) # ACCEPTED

                        config[int(t),int(s)] = [fill,fill] # remove particle at current, and go again at previous
                        if (i1==fill or config[int((t-1)%k),int(i1),1]==fill):
                            break

                        # Begin again at the previous site from the world line you collided with
                        s1 = i1 #Store previous site index
                        i2 = config[int((t-1)%k),int(s1),0] #Grab previous of previous site
                        t, s, i = (t-1)%k, s1, i2
                        continue # Re-enter the algorithm at the top

                    else: #REJECTED # still remove particle at current, but move forward from previous
                        p = np.random.random()
                        a = 1
                        config[int(t),int(s)] = [fill,fill] # I'm assuming this removes the particle from the site (t,s) ## I think you meant delSite(t,s) ?

                        if (i1==fill):
                            break

                        if  p < remain_thresh: #(1-2*self.d*self.t*self.e) # REMAIN AT SAME SITE ("same" meaning previous of where we proposed from)
                            s1 = i1 #Store previous site
                            i2 = config[int((t-1)%k),int(s1),0] #Get previous of previous site (has to stay the same for continuity)
                            config[int((t-1)%k),int(s1)]=[i2,0] #Reset direction of previous timestep
                            s, i = s1, i1
                            continue # Re-enter the algorithm at the top
                        
                        else: # MOVE TO DIFFERENT SITE
                            s1 = i1 #Store previous site
                            i2 = config[int((t-1)%k),int(s1),0] #Get Previous
                            config[int((t-1)%k),int(s1)]=[i2,d] #Reset direction of previous timestep 
                            s1 = dmap[int(s1)][int(d)] #Get new site you are moving to in current timestep t
                            d  = directions[np.random.randint(0,2*dim)]
                            s, i = s1, i1
                            continue # Re-enter the algorithm at the top

        # Return updated configuration
        return config

#----------------------------------------------------------#
        
class Worm:
    
    def __init__(self,t,mu,e,k,N,l):
        """
        t   - t paramter of hamiltonian
        mu  - mu parameter of hamiltonian
        e   - timestep size
        k   - # of unique time steps in world line
        N   - array containing # of lattice sites along each spatial axis
        l   - lattice object
        """
        self.t   = t
        self.mu  = mu
        self.e   = e
        self.k   = k
        self.N   = N
        self.d   = len(N) # Spatial dimension
        self.l   = l

        # Create numba dictionary for grabbing neighbors
        dmap = l.neighbors
        self.dmap = Dict()
        for s in dmap:
            submap = Dict.empty(
                                key_type=types.int64,
                                 value_type=types.int64
                                )
            for d in dmap[s]:
                submap[d] = dmap[s][d]
            self.dmap[s] = submap

    def apply(self,seedConfig):
        """
        Run worm algorithm on seedConfig and return a new updated config WITH numba speedup of entire algorithm.
        """
        config = seedConfig.clone()
        
        # Get random starting time, site, and direction
        t = np_rand_randint(0,self.k)
        s = np_rand_randint(0,len(self.l.sites))
        i = config.fill #NOTE: Going off of Shailesh's discussion with Mingru in Nov. 10th office hrs we can just label all the world lines the same
        d = self.l.directions[np_rand_randint(0,2*self.d)]
        a = 1

        config.config = apply_(
            self.t, self.mu, self.e, self.k, self.d,
            t, s, i, d, a,
            config.config.copy(),
            self.dmap,
            np.array(self.l.directions),
            config.fill
        )
        
        return config

        
    def apply_no_numba(self,seedConfig):
        """
        Run worm algorithm on seedConfig and return a new updated config WITHOUT numba speedup of entire algorithm.
        """
        config = seedConfig.clone()
        
        # Get random starting time, site, and direction
        t = np_rand_randint(0,self.k)
        s = np_rand_randint(0,len(self.l.sites))
        i = config.fill #NOTE: Going off of Shailesh's discussion with Mingru in Nov. 10th office hrs we can just label all the world lines the same
        d = self.l.directions[np_rand_randint(0,2*self.d)]
        a = 1

        # Pulled these out to help speed up the loop
        forward_thresh  = np.exp(self.mu*self.e)
        backward_thresh = np.exp(-self.mu*self.e)
        remain_thresh   = (1-2*self.d*self.t*self.e)

        while True:
            """
            t - time slice #
            s - site #
            i - world line index
            d - direction # (for next step, set to 0 if staying on same site)
            a - arrow
            """
            
            # NO PARTICLE AT SITE
            if config.checkSite(t,s):

                p = np_rand_random() #NOTE: Jit wrapped function # get random float in [0,1)
                if a == 1: # check arrow direction

                    # PROPOSE FORWARD
                    if p < forward_thresh: # ACCEPTED #np.exp(self.mu*self.e)         
                        p = np_rand_random() #NOTE: Jit wrapped function

                        if  p < remain_thresh: # REMAIN AT SAME SITE #(1-2*self.d*self.t*self.e)
                            config.setSite(t,s,i,0)
                            t, i = (t+1)%self.k, s
                            continue # Re-enter the algorithm at the top

                        else: # MOVE TO DIFFERENT SITE
                            config.setSite(t,s,i,d) #NOTE: This had the same typo as above: changed (t,s,d,i) to (t,s,i,d)
                            i = s
                            s = self.l.getNeighbor(s,d)
                            d = self.l.directions[np_rand_randint(0,2*self.d)]
                            t = (t+1)%self.k
                            continue # Re-enter the algorithm at the top

                    else: # PROPOSE FORWARD REJECTED # just flip direction/arrow to back and not move in this step
                        a = -1
                        continue # Re-enter the algorithm at the top
                elif a == -1:
                    break

            # ALREADY PARTICLE AT SITE
            else:
                i1, d1 = config.getSite(t,s)
                if a == 1: # if we got here from moving forward   # basing this on slide 46, where he just flips arrow and then seems to let backwards proposal take care of it                                                             
                    a=-1
                    config.setSite(t,s,i,d1) #NOTE: Reset the "previous" entry at the colliding site so that the world line is continuous
                    if (i1==config.fill or config.checkSite((t-1)%self.k,i1)):#NOTE: DEBUGGING have an or for if you search previous time slice at requested previous site and find no particle
                        break

                    # Begin again at the previous site from the world line you collided with
                    s1 = i1 #Store previous site index
                    i2 = config.getPrevious((t-1)%self.k,s1) #Grab previous of previous site
                    if (i2==config.fill or config.checkSite((t-2)%self.k,i2)): #NOTE: Need this break here because you have to eat up the world line to the next time slice and if it's the end just end, should be taken care of below but apparently not
                        config.delSite((t-1)%self.k,s1)
                        break
                    t, s, i = (t-1)%self.k, s1, i2
                    continue # Re-enter the algorithm at the top

                elif a == -1: # if we got here from moving backwards, then I think we just propose backward again
                    p = np_rand_random() # jit wrapped function # PROPOSE BACKWARDS
                    if p < backward_thresh: #np.exp(-1*self.mu*self.e) # ACCEPTED
                        config.delSite(t,s) # remove particle at current, and go again at previous
                        if (i1==config.fill or config.checkSite((t-1)%self.k,i1)): #NOTE: Moved this to be after config.delSite(t,s) not sure but I think we still delete the site here
                            break

                        # Begin again at the previous site from the world line you collided with
                        s1 = i1 #Store previous site index
                        i2 = config.getPrevious((t-1)%self.k,s1) #Grab previous of previous site
                        t, s, i = (t-1)%self.k, s1, i2
                        continue # Re-enter the algorithm at the top

                    else: #REJECTED
                        p = np_rand_random() # still remove particle at current, but move forward from previous
                        a = 1
                        config.delSite(t,s)

                        if (i1==config.fill):
                            break

                        # REMAIN AT SAME SITE ("same" meaning previous of where we proposed from)
                        if  p < remain_thresh: #(1-2*self.d*self.t*self.e)
                            s1 = i1 #Store previous site
                            i2 = config.getPrevious((t-1)%self.k,s1) #Get previous of previous site (has to stay the same for continuity)
                            config.setSite((t-1)%self.k,s1,i2,0) #Reset direction of previous timestep
                            s, i = s1, i1 #run again, from the same time position and arrow forward (slide 45)
                            continue # Re-enter the algorithm at the top
                        # MOVE TO DIFFERENT SITE
                        else:
                            s1 = i1 #Store previous site
                            i2 = config.getPrevious((t-1)%self.k,s1)
                            config.setSite((t-1)%self.k,s1,i2,d) #Reset direction of previous timestep 
                            s1 = self.l.getNeighbor(s1,d) #Get new site you are moving to in current timestep t
                            d  = self.l.directions[np_rand_randint(0,2*self.d)]
                            s, i = s1, i1
                            continue # Re-enter the algorithm at the top

        # Return updated configuration
        return config

    
class FileManager:
    
    def __init__(self,fileName,maxEntries=1e4):
        """
        See https://numpy.org/devdocs/user/how-to-io.html for more detail.
        Arguments:
            fileName   - pretty self-explanatory
            maxEntries - ditto
        """
        self.baseName   = fileName
        self.fileName   = fileName
        self.maxEntries = maxEntries
        self.counter    = 0
        # if self.fileName != "": self.f = open(self.fileName,'ab')
        self.f = None
        
    def switchOutFile(self):
        if self.counter != 0: self.f.close()
        self.fileName = '.'.join([
            str(s) for s in self.baseName.split('.')[:-1]]) + "_" + str(int(self.counter//self.maxEntries)) + "." + self.baseName.split('.')[-1]
        self.f = open(self.fileName,'ab')

    def append(self,config,lattice,beta,t,mu,obs=["N","E","ET","W","W2"],fmt='%f',delimiter='\t'):
        if self.counter % self.maxEntries == 0: self.switchOutFile()
        header = ''
        if self.counter % self.maxEntries == 0: header = delimiter.join([o for o in obs])
        else: header = ''
        np.savetxt(self.f, np.c_[[config.getObs(lattice,beta,t,mu,obs)]],
                   header=header, fmt=fmt, delimiter=delimiter)
        self.counter += 1

    def open(self):
        self.f = open(self.fileName,'ab')

    def close(self):
        self.f.close()
        
    def read(self,fileNames,dtype=np.float32,delimiter="\t",usecols=None):
        """
        fileNames - pretty self-explanatory
        dtype     - ditto
        delimiter - ditto
        usecols   - default None selects all columns, pass a tuple of ints to select specific columns, e.g. (1,4,5) for 2nd, 5th and 6th columns
        """
        arr = None
        flag = 0
        for fileName in fileNames:
            f = open(fileName,'r')
            if flag==0:
                arr  = np.loadtxt(f, dtype=np.float32, delimiter=delimiter, usecols=usecols)
                flag = 1
            else: arr = np.concatenate((arr,np.loadtxt(f, dtype=np.float32, delimiter=delimiter, usecols=usecols)))
            f.close()
        return arr


class Generator:
    
    def __init__(self,e,t=1.0,beta=12.0,mu=1.4,N=[2,2,2],seed=17,fileName="out.txt",maxEntries=1e4):
        """
        Arguments:
            e          - timestep size
            t          - t parameter of Hamiltonian
            mu         - mu parameter of Hamiltonian (chemical potential)
            beta       - beta parameter for ensemble averages
            N          - array containing # of lattice sites along each spatial lattice
            seed       - seed integer for numpy built-in random # generator
            fileName   - basename for output files
            maxEntries - max # of configuration entries per output file
        """
        
        self.t    = t
        self.mu   = mu
        self.e    = e
        self.beta = beta
        self.k    = beta/e # of timesteps
        self.N    = N
        
        self.seedConfig  = Configuration(self.k,self.N)
        self.fileManager = FileManager(fileName,maxEntries=maxEntries)
        self.lattice     = Lattice(N)
        self.algorithm   = Worm(t,mu,e,self.k,N,self.lattice)

    def seed(self,seed):
        np.random.seed(seed)
        
    def generate(self,seedConfig):
        """
        Apply worm algorithm to config and return new config
        """
        return self.algorithm.apply(seedConfig)
    
    def loop(self,nIterations,obs=["N","E","ET","W","W2"]):
        """
        Generate a given number of configurations and write configuration observables to file
        """
        self.fileManager.open()
        config = self.seedConfig.clone()
        for n in tqdm(range(nIterations)):
            config = self.generate(config)
            self.fileManager.append(config,self.lattice,self.beta,self.t,self.mu,obs)
        self.fileManager.close()
            