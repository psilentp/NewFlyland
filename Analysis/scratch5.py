import expansion_lib as expl
import pylab as plb
from scipy.signal import hilbert
import numpy as np
import neo
import quantities as pq
#def get_phase(h1,h2):
#    return angle(hilbert(expl.get_low_filter(h1+h2,500))))
    
flynum = 14
#flynum = 6 ##fly for first figure (sweep 14)
#[[ 1.63812081  1.63159965]
# [-0.38874933 -0.38720176]]
dataroot = '/Users/psilentp/Dropbox/Data/LeftRight/'

cluster_dict = {2:2,
                3:2,
                4:2,
                5:np.array([[ 1.29689704,1.38142633],[-0.3490147,-0.37176281]]),
                6:np.array([[-0.75975139,1.07104702],[ 0.4440261,-0.62595848]]),
                7:np.array([[ 1.25865512,1.26488943],[-0.52354824,-0.52614145]]),
                8:2,
                9:2,
                10:2,
                11:np.array([[ 0.41559677,1.23608377],[-0.16682716,-0.49618372]]),
                12:2,
                13:2,
                14:np.array([[-0.72394844,0.6905216 ],[ 0.50694304,-0.48353598]]),
                15:2,
                16:2,
                17:2,
                18:2,
                19:2,
                20:2}

fly = expl.FlyRecord(flynum,0,dataroot)

numtrials = 15

def plot_trial_spktrns(indx = 5):
    print('plot spike rasters for expansion from left vs right -1 to 0.2 sec')
    spiketrains = list()
    for x in range(numtrials):
        #spiketrains.append(fly.extract_trial_spikes(2,indx,x,-1,0.2))
        try:
            spiketrains.append(fly.extract_trial_spikes(2,indx,x,-1,0.2))
        except IndexError:
            pass    
    plb.figure(figsize= (8,20))
    ax0 = plb.subplot(4,1,1)
    behv = expl.get_signal_mean([expl.ts(s,-1,0.2) for s in fly[2,indx,:numtrials,'L_m_R']])
    stim = expl.get_signal_mean([expl.expan_transform(expl.ts(s,-1,0.2)) for s in fly[2,indx,:numtrials,'Xpos']])
    plb.plot(behv.times,stim)
    
    ######plot behavior#######
    ax1 = plb.subplot(4,1,2,sharex = ax0)
    plb.plot(behv.times,behv)
    
    ######plot raster########
    ax2 = plb.subplot(4,1,3, sharex = ax0)
    wvfm_list = list()
    for trial,st in enumerate(spiketrains):
        #st = st[2:-3]
        #print 'stacking'
        wvfm_list.extend([{'trial':trial,'time':stime,'waveform':x,'phase':p} for stime,x,p in zip(st[2:-2], st.waveforms[2:-2], st.annotations['phases'][2:-2])])

    datamtrx = np.vstack([np.array(x['waveform']) for x in wvfm_list])
    #print 'clustering'
    M = cluster_dict[flynum]
    idx,features,U = expl.sort_spikes(datamtrx,M)
    for i,wvfm in zip(idx,wvfm_list):
        wvfm['id'] = i
    
    inc_trains = list()
    filtered = [w for w in wvfm_list if w['id'] == 1]
    #print len(filtered), 0.5*len(wvfm_list)
    if len(filtered) < 0.5*len(wvfm_list):
        filtered = [w for w in wvfm_list if (w['id'] == 0)]
    #print len(filtered), 0.5*len(wvfm_list)
    for trial,st in enumerate(spiketrains):
        st = st[2:-2]
        trial_spikes = [w for w in filtered if (w['trial'] == trial)]
        times = [w['time'] for w in trial_spikes]
        waveforms = [w['waveform'] for w in trial_spikes]
        phases = np.array([w['phase'] for w in trial_spikes])
        inc_trains.append(neo.SpikeTrain(times*pq.Quantity(1,'s'),
                                st.t_stop,
                                sampling_rate = st.sampling_rate,
                                waveforms = waveforms,
                                left_sweep = st.left_sweep,
                                t_start = st.t_start,
                                phases = phases))
        
    expl.phase_raster([np.array(trn) for trn in inc_trains],[trn.annotations['phases'] for trn in inc_trains])
    
    #####Phase Trace######
    ax3 = plb.subplot(4,1,4, sharex = ax0)
    [plb.plot(np.array(trn),trn.annotations['phases']/np.pi,'o',color = 'k',alpha = 0.2) for trn in inc_trains]
    
    
    ax0.set_xbound(-1,0.2)
    ax0.set_ybound(0,100)
    ax1.set_ybound(-7,7)
    ax3.set_ybound(0,2)
    plb.show()
    return idx,wvfm_list,inc_trains
    
def plot_single_sweep_intro(trial_num,inc_trains,indx):
    plb.figure(figsize = (20,15))
    ephys_sweep = expl.ts(fly[2,indx,trial_num,'AMsysCh1'][0],-1,0.2)
    left_wing = expl.ts(fly[2,indx,trial_num,'LeftWing'][0],-1,0.2)
    right_wing = expl.ts(fly[2,indx,trial_num,'RightWing'][0],-1,0.2)
    La,Ra = expl.get_wbas(left_wing,right_wing)
    phases = expl.get_phase_trace(left_wing,right_wing)
    stim = expl.expan_transform(expl.ts(fly[2,indx,trial_num,'Xpos'][0],-1,0.2))
    
    ###########Stim#########
    ax0 = plb.subplot(4,1,1)
    plb.plot(stim.times,stim,color = 'k',alpha = 0.5)
    ax0.set_ybound(0,100)
    
    ##########Behavioir###########
    ax1 = plb.subplot(4,1,2,sharex = ax0)
    plb.plot(left_wing.times,left_wing,color = (0.7,0.4,0.2),alpha = 0.8)
    plb.plot(right_wing.times,right_wing,color = (0.2,0.4,0.7),alpha = 0.8)    
    
    #plb.plot(left_wing.times,La,color = (0.7,0.4,0.2),alpha = 0.8)
    #plb.plot(right_wing.times,Ra,color = (0.2,0.4,0.7),alpha = 0.8)

    #plb.plot(right_wing.times,La-Ra,color = 'r',alpha = 0.8)
    
    ###########Phases#############
    ax2 = plb.subplot(4,1,3,sharex = ax0)
    plb.plot(left_wing.times,phases/np.pi,color = 'k',alpha = 0.5)
    tst_sweep = fly.extract_trial_spikes(2,indx,trial_num,-1,0.2)
    colors = [plb.cm.jet(c/(2*np.pi)) for c in inc_trains[trial_num].annotations['phases']]
    plb.scatter(np.array(inc_trains[trial_num].times),np.array(inc_trains[trial_num].annotations['phases']/np.pi),c = colors,s = 80)
    #plb.plot(np.array(inc_trains[trial_num].times),np.array(inc_trains[trial_num].annotations['phases']),'o',color = 'r')
    #plb.plot(np.array(tst_sweep.times),np.array(tst_sweep.annotations['phases']),'o',color = 'g')
    
    ############Spikes###########
    ax3 = plb.subplot(4,1,4,sharex = ax0)
    plb.plot(ephys_sweep.times,ephys_sweep,color = 'k');
    [plb.plot(x.times,x,color = plb.cm.jet(c/(2*np.pi)),lw =4,alpha =0.5) for x,c in zip(inc_trains[trial_num].waveforms,inc_trains[trial_num].annotations['phases'])]
    ax0.set_xbound(-0.2,0.05)
    
    ax0.set_ylabel(u'Stimulus extent (\u00B0)')
    ax1.set_ylabel(u'Wingbeat signal (V)')
    ax2.set_ylabel(u'Stroke phase (rad)')
    ax3.set_ylabel(u'Electrode (\u00B5V)')
    
    ax3.set_xlabel(u'Time (s)')
    
    ax2.set_yticks([0,1,2])
    ax2.set_yticklabels([u'0',u'\u03C0',u'2\u03C0'])
    
    ticklabelpad = plb.rcParams['xtick.major.pad']
    #ax3.annotate('XLabel', xy=(-0.15,0.5), xytext=(5, -ticklabelpad), ha='center', va='center',rotation = 90,
    #        xycoords='axes fraction', textcoords='offset points')
    #GREEK SMALL LETTER PI
    #Unicode: U+03C0, UTF-8: CF 80
    #MICRO SIGN
    #Unicode: , UTF-8: C2 B5
    #DEGREE SIGN
    #Unicode: U+00B0, UTF-8: C2 B0 
    plb.show()
    
