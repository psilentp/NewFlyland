import expansion_lib as expl
import pylab as plb
from scipy.signal import hilbert
import numpy as np
import neo
import quantities as pq
#def get_phase(h1,h2):
#    return angle(hilbert(expl.get_low_filter(h1+h2,500))))
    
dataroot = '/Users/psilentp/Dropbox/Data/LeftRight/'

fly = expl.FlyRecord(2,0,dataroot)

numtrials = 13

def plot_trial_spktrns(indx = 5):
    print('plot spike rasters for expansion from left vs right -1 to 0.2 sec')
    spiketrains = list()
    for x in range(numtrials):
        try:
            spiketrains.append(fly.extract_trial_spikes(2,indx,x,-1,0.2))
        except IndexError:
            pass
    plb.figure()
    ax0 = plb.subplot(3,1,1)
    #plot behavior
    behv = expl.get_signal_mean([expl.ts(s,-1,0.2) for s in fly[2,indx,:numtrials,'L_m_R']])
    plb.plot(behv.times,behv)
    #plot raster
    ax1 = plb.subplot(3,1,2, sharex = ax0)
    inc_trains = list()
    for st in spiketrains: 
        idx,features,U = expl.sort_spikes(np.array(st.waveforms[:-3]))
        Umean = np.mean(U[:,0])
        Ustd = np.std(U[:,0])
        #criterion = U[:,0]>(Umean-50*Ustd)
        #criterion = criterion & (np.diff(np.array(st))>0.003)[:-2] 
        criterion = (np.diff(np.array(st))>0.003)[:-2]      
        times = [st[i] for i,x in enumerate(criterion) if x]
        waveforms = [st.waveforms[i] for i,x in enumerate(criterion) if x]
        phases = np.array([st.annotations['phases'][i] for i,x in enumerate(criterion) if x])
        print len(st),len(times)
        inc_trains.append(neo.SpikeTrain(times*pq.Quantity(1,'s'),
                                st.t_stop,
                                sampling_rate = st.sampling_rate,
                                waveforms = waveforms,
                                left_sweep = st.left_sweep,
                                t_start = st.t_start,
                                phases = phases))
                                
                                
    expl.phase_raster([np.array(trn) for trn in inc_trains],[trn.annotations['phases'] for trn in inc_trains])
    ax1 = plb.subplot(3,1,3, sharex = ax0)
    [plb.plot(np.array(trn),trn.annotations['phases'],'o',color = 'k',alpha = 0.2) for trn in inc_trains]
    plb.show()
    return inc_trains

#plot(sweep.times,sweep,color = 'k');[plot(x.times,x,color = cm.jet(c/(2*pi)),lw =2,alpha =0.7) for x,c in zip(inc_trains[0].waveforms,inc_trains[0].annotations['phases'])]
#plot(sweep.times,sweep);[plot(x.times,x,color = 'r',lw =4,alpha =0.5) for x in inc_trains[0].waveforms]
"""
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
"""

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