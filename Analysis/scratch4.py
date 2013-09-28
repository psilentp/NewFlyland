import numpy as np
import pylab as plb

def plot_lowspikes(spiketrains):
    for st in spiketrains:
        lowind = np.argwhere(st.annotations['phases']>4)
        [plb.plot(st.waveforms[x], color = 'k',alpha = 0.5) for x in lowind[:20]]
    