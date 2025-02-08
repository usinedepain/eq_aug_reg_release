import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys


"""
        Code for producing graphics as in the paper.
"""
gamma = sys.argv[1]
NMBR_RUNS = sys.argv[2]

mpl.rcParams['text.usetex'] = True


# load the results from file
file = os.path.join('results','0_'+gamma+'_0.npz')
result = np.load(file)


# arrays for the projection error for the nominal and augmented model
proj = np.zeros((NMBR_RUNS, 10*len(result['diff']),3))
aproj = np.zeros((NMBR_RUNS, 10*len(result['diff']),3))


for k in range(10):
    for run in range(NMBR_RUNS):

        # go through the runs and load the results into the arrays
        file = os.path.join('results',str(run)+'_'+gamma+'_'+str(k)+'.npz')
        result = np.load(file)

        proj[run,len(result['diff'])*k: len(result['diff'])*(k+1),:] = result['projerror']
        aproj[run,len(result['diff'])*k: len(result['diff'])*(k+1),:] =result['aprojerror']



# calculate the medians
med= np.median(proj[:,:,:].sum(-1),axis=0)

amed= np.median(np.sqrt(aproj[:,:,:].sum(-1)),axis=0)

#plot the figures and save them
plt.plot(np.sqrt(proj[:,:,:].sum(-1)).T,alpha=.05,c='blue')
plt.plot(np.sqrt(med),c='blue',label='Nom')
plt.plot(np.sqrt(aproj[:,:,:].sum(-1)).T,alpha=.05,c='red')
plt.plot(amed,c='red',label='Aug', linestyle ='dashed')
plt.title('Projection error, $\gamma$=1e'+gamma)
plt.ylim(5e-4,3e0)
plt.legend(fontsize=25)
plt.yscale('log')
plt.savefig('proj_'+gamma)