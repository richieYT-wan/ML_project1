import numpy as np
import matplotlib.pyplot as plt
from implementations import *

"""Visualization of the feature distributions data per cluster."""

def visualization_hist_cluster(t0,t1,t2,t3,header):
        
    fig, axes = plt.subplots(10,3,figsize=(10,20),sharex =True,sharey =True,)
    a = axes.ravel()
    
    for idx,ax in enumerate(a[:]):        
        ax.hist(t0[:,idx],bins=20,stacked=True, histtype='stepfilled',alpha=0.5)
        ax.hist(t1[:,idx],bins=20,stacked=True, histtype='stepfilled',alpha=0.5)
        ax.hist(t2[:,idx],bins=20,stacked=True, histtype='stepfilled',alpha=0.5)
        ax.hist(t3[:,idx],bins=20,stacked=True, histtype='stepfilled',alpha=0.5)
        #ax.set_adjustable(self, adjustable, share=False)
        ax.autoscale()

        ax.set_title(header[idx+2])
    plt.tight_layout() 
    plt.show()

