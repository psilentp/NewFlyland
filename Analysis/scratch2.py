import expansion_lib as expl
import pylab as plb
from scipy.signal import hilbert
import numpy as np

#def get_phase(h1,h2):
#    return angle(hilbert(expl.get_low_filter(h1+h2,500))))
    
dataroot = '/Users/psilentp/Dropbox/Data/LeftRight/'

fly = expl.FlyRecord(2,0,dataroot)

numtrials = 13
print('plot spike rasters for expansion from left vs right -1 to 0.2 sec')
spiketrains = list()
for x in range(numtrials):
    try:
        spiketrains.append(fly.extract_trial_spikes(2,5,x,-1,0.2))
    except IndexError:
        pass
            
#spiketrains = [fly.extract_trial_spikes(2,5,x,-1,0.2) for x in range(numtrials)]
plb.figure()
ax0 = plb.subplot(2,1,1)
#plot behavior
behv = expl.get_signal_mean([expl.ts(s,-1,0.2) for s in fly[2,5,:numtrials,'L_m_R']])
plb.plot(behv.times,behv)
#plot raster
ax1 = plb.subplot(2,1,2, sharex = ax0)
expl.phase_raster([np.array(trn) for trn in spiketrains],[trn.annotations['phases'] for trn in spiketrains])


###########################
plb.figure()
spiketrains2 = [fly.extract_trial_spikes(2,11,x,-1,0.2) for x in range(numtrials)]
plb.figure()
ax0 = plb.subplot(2,1,1)
#plot behavior
behv2 = expl.get_signal_mean([expl.ts(s,-1,0.2) for s in fly[2,11,:numtrials,'L_m_R']])
plb.plot(behv.times,behv2)

#plot raster
ax1 = plb.subplot(2,1,2, sharex = ax0)
expl.phase_raster([np.array(trn) for trn in spiketrains2],[trn.annotations['phases'] for trn in spiketrains2])


plb.show()
"""
sweeps2 = [expl.ts(s,-1,0.2) for s in fly[2,11,:3,'AMsysCh1']]
print('calculating spiketrains for second set of sweeps')
spiketrains2 = [expl.get_spiketrain(s) for s in sweeps2]
behv2 = expl.get_signal_mean([expl.ts(s,-1,0.2) for s in fly[2,11,:3,'L_m_R']])

plb.figure()
ax0 = plb.subplot(2,1,1)
#plot behavior
behv2 = expl.get_signal_mean([expl.ts(s,-1,0.2) for s in fly[2,11,:3,'L_m_R']])
plb.plot(behv2.times,behv2)

#plot raster
ax1 = plb.subplot(2,1,2, sharex = ax0)
expl.raster(spiketrains2)

plb.show()
#plot(angle(hilbert(expl.get_low_filter(huchens2+huchens,500))))
"""