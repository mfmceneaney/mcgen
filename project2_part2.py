#--------------------------------------------------#
# Script to compute chi2 for Project 2 part 2      #
#                                                  #
#--------------------------------------------------#
from analysislib import *
from mclib import FileManager

# Initialize 
fm = FileManager("")
directory = "../mcgen_out/mcgen_e0.01_N20_mu1.0_b0.75-1.1/"
nFiles = 1 # Per mu value or which ever parameter you choose to plot
beta   = [fl/100 for fl in range(75,111,1)]
obs    = ["N","E","ET","W","W2","SHops","THops"]
names  = {obs[i]:i for i in range(len(obs)) }
obs    = ["W2"] # reset
data   = {obs[i]:[] for i in range(len(obs)) }
#NOTE: Do not use -1 for i2 since function is jit wrapped and can't handle negative indexing
i1, i2, step, nsigma, trail = 999, 10000, -1, 2, 1000

N = 20#10000 - i1 # Know already so don't bother computing

# Loop over input files and compile data points
for b in beta:
    data_e = fm.read([directory+f"/mcgen_e0.01_N20_mu1.0_b{b:.2f}_"+str(i)+".txt" for i in range(nFiles)])
    for ob in obs:
        data[ob].append(statistics(data_e,names[ob],i1,i2,step=step,nsigma=nsigma,trail=trail))
        plot_blocks(data_e,ob,names[ob],i1,i2,step=step,nsigma=nsigma,trail=trail)

# Plot <n> as function of beta
y    = 1/N*np.divide(np.array(data["W2"])[:,0],beta)
yerr = 1/N*np.divide(np.array(data["W2"])[:,1],beta)
plot_obs(np.array(beta),y,yerr,xlabel=r'$\beta$',ylabel="$\chi_{\omega}$")

# Print data for tabulation
print(beta)
print("----------")
print(y)
print(yerr)

# Print data for tabulation in latex
print('\\\\\n'.join([
    " & ".join([
    f"{np.real(beta[i]):.2f}",
    f"{np.real(y[i]):.6f} $\pm$ {np.real(yerr[i]):.6f}"
])
    for i in range(len(beta))
]))
