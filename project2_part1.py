#--------------------------------------------------#
# Script to compute n ave for Project 2 part 1     #
#                                                  #
#--------------------------------------------------#
from analysislib import *
from mclib import FileManager
import numpy as np

# Initialize 
fm = FileManager("")
directory = "../drop/mcgen_new/mcgen_e0.01_N4_mu0.00-0.50_b100.0/"
nFiles = 1 # Per mu value or which ever parameter you choose to plot
mu  = [fl/100 for fl in range(0,51,1)]
obs = ["N"] #["N","E","ET","W","W2","SHops","THops"]
names = {obs[i]:i for i in range(len(obs)) }
data = {obs[i]:[] for i in range(len(obs)) }
#NOTE: Do not use -1 for i2 since function is jit wrapped and can't handle negative indexing
i1, i2, step, nsigma, trail = 999, 10000, -1, 2, 1000 

# Loop over input files and compile data points
for m in mu:
    data_e = fm.read([directory+f"/mcgen_e0.01_N4_mu{m:.2f}_b100.0_"+str(i)+".txt" for i in range(nFiles)])
    for ob in obs:
        helper = statistics(data_e,names[ob],i1,i2,step=step,nsigma=nsigma,trail=trail)
        data[ob].append(helper)
        plot_blocks(data_e,ob,names[ob],i1,i2,step=step,nsigma=nsigma,trail=trail)

# Plot <n> as function of mu
y = np.array(data["N"])[:,0]
yerr = np.array(data["N"])[:,1]
plot_obs(np.array(mu),y,yerr,xlabel="$\mu$",ylabel="$<n>_\epsilon$") #TODO: Plot together with exact results

# Print data for tabulation
print(mu)
print("----------")
print(y)
print(yerr)

print('\\\\\n'.join([
    " & ".join([
    f"{np.real(mu[i]):.2f}",
    f"{np.real(y[i]):.5f} $\pm$ {np.real(yerr[i]):.5f}"
])
    for i in range(len(mu))
]))

# Try to save in latex format for table
a = np.array([mu,y,yerr])
np.savetxt("aven.csv", a, delimiter=' & ', fmt='%2.2e', newline=' \\\\\n')
