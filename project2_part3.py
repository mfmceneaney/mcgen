#--------------------------------------------------#
# Script to compute chi_omega for Project 2 part 3 #
#                                                  #
#--------------------------------------------------#
from analysislib import *
from mclib import FileManager

# Initialize 
fm = FileManager("")
directory1 = "../mcgen_out/mcgen_e0.01_N16_mu1.0_b0.88-0.95/"
directory2 = "../mcgen_out/mcgen_e0.01_N20_mu1.0_b0.88-0.95/"
directory3 = "../mcgen_out/mcgen_e0.01_N24_mu1.0_b0.88-0.95/"
nFiles = 1 # Per mu value or which ever parameter you choose to plot
beta   = [fl/100 for fl in range(88,96,1)]
obs    = ["N","E","ET","W","W2","SHops","THops"]
names  = {obs[i]:i for i in range(len(obs)) }
obs    = ["W2"]  # reset
data1   = {obs[i]:[] for i in range(len(obs)) }
data2   = {obs[i]:[] for i in range(len(obs)) }
data3   = {obs[i]:[] for i in range(len(obs)) }
#NOTE: Do not use -1 for i2 since function is jit wrapped and can't handle negative indexing
i1, i2, step, nsigma, trail = 999, 10000, -1, 2, 1000

N1, N2, N3 = 16, 20, 24

# Loop over input files and compile data points for N=16
for b in beta:
    data_e = fm.read([directory1+f"/mcgen_e0.01_N16_mu1.0_b{b:.2f}_"+str(i)+".txt" for i in range(nFiles)])
    for ob in obs:
        data1[ob].append(statistics(data_e,names[ob],i1,i2,step=step,nsigma=nsigma,trail=trail))
        plot_blocks(data_e,ob,names[ob],i1,i2,step=step,nsigma=nsigma,trail=trail)

# Loop over input files and compile data points for N=20
for b in beta:
    data_e = fm.read([directory2+f"/mcgen_e0.01_N20_mu1.0_b{b:.2f}_"+str(i)+".txt" for i in range(nFiles)])
    for ob in obs:
        data2[ob].append(statistics(data_e,names[ob],i1,i2,step=step,nsigma=nsigma,trail=trail))
        # plot_blocks(data_e,ob,names[ob],i1,i2,step=step,nsigma=nsigma,trail=trail)

# Loop over input files and compile data points for N=24
for b in beta:
    data_e = fm.read([directory3+f"/mcgen_e0.01_N24_mu1.0_b{b:.2f}_"+str(i)+".txt" for i in range(nFiles)])
    for ob in obs:
        data3[ob].append(statistics(data_e,names[ob],i1,i2,step=step,nsigma=nsigma,trail=trail))
        # plot_blocks(data_e,ob,names[ob],i1,i2,step=step,nsigma=nsigma,trail=trail)

# Plot <n> as function of beta
y1    = 1/N1*np.divide(np.array(data1["W2"])[:,0],beta)
yerr1 = 1/N1*np.divide(np.array(data1["W2"])[:,1],beta)
y2    = 1/N2*np.divide(np.array(data2["W2"])[:,0],beta)
yerr2 = 1/N2*np.divide(np.array(data2["W2"])[:,1],beta)
y3    = 1/N3*np.divide(np.array(data3["W2"])[:,0],beta)
yerr3 = 1/N3*np.divide(np.array(data3["W2"])[:,1],beta)
ys = [y1,y2,y3]
yerrs = [yerr1,yerr2,yerr3]
l1 , l2 , l3 = "N=16", "N=20", "N=24"
plot_obs_together(np.array(beta),ys,yerrs,xlabel=r'$\beta$',ylabel="$\chi_{\omega}$",labels=[l1,l2,l3])

# Print data for tabulation in latex
print('\\\\\n'.join([
    " & ".join([
    f"{np.real(beta[i]):.2f}",
    f"{np.real(y1[i]):.6f} $\pm$ {np.real(yerr1[i]):.6f}",
    f"{np.real(y2[i]):.6f} $\pm$ {np.real(yerr2[i]):.6f}",
    f"{np.real(y3[i]):.6f} $\pm$ {np.real(yerr3[i]):.6f}"
])
    for i in range(len(beta))
]))
