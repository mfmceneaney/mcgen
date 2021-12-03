#--------------------------------------------------#
# Script to plot and fit results for Project 1     #
# part 2                                           #
#--------------------------------------------------#
from analysislib import *

fm = FileManager("")
obs = ["N","E","ET","W","W2","SHops","THops"]
names = {obs[i]:i for i in range(len(obs)) }
nFiles = 10
obs = ["N","E","ET","W","W2"]#["N","E","ET","W","W2","SHops","THops"]
ep  = [0.03,0.02,0.01,0.008,0.005,0.002,0.001]
data = {obs[i]:[] for i in range(len(obs)) }
i1, i2, step, nsigma, trail = 9999, -1, -1, 2, 1000
for e in ep:
    data_e = fm.read(["mcgen/mcgen_e"+str(e)+"/mcgen_"+str(i)+".txt" for i in range(nFiles)])
    for ob in obs:
        data[ob].append(statistics(data_e,names[ob],i1,i2,step=step,nsigma=nsigma,trail=trail))
        plot_blocks(data_e,ob,names[ob],i1,i2,step=step,nsigma=nsigma,trail=trail)

yn = np.array(data["N"])[:,0]
ynerr = np.array(data["N"])[:,1]
ye = np.array(data["ET"])[:,0]
yeerr = np.array(data["ET"])[:,1]
plot_fit_obs(np.array(ep),yn,ynerr,ylabel="$<n>_\epsilon$")
plot_fit_obs(np.array(ep),ye,yeerr,ylabel="$<\~{e}>_\epsilon$")

# Print data for tabulation in latex
print('\\\\\n'.join([
    " & ".join([
    f"{ep[i]}",
    f"{yn[i]:.4f}  $\pm$ {ynerr[i]:.4f}",
    f"{ye[i]:.4f}  $\pm$ {yeerr[i]:.4f}"
])
    for i in range(len(ep))
]))