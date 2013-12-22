
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
    ax0 = plb.axes([0.10,0.85,0.85,0.10])
    ax1 = plb.axes([0.10,0.60,0.85,0.20],sharex = ax0)
    ax2 = plb.axes([0.10,0.35,0.85,0.20],sharex = ax0)
    ax3 = plb.axes([0.10,0.10,0.85,0.20],sharex = ax0)
    
    plb.axes(ax0)#plb.subplot(4,1,1)
    LeftWings = [expl.ts(s,-1,0.2) for s in fly[2,indx,:numtrials,'LeftWing']]
    RightWings = [expl.ts(s,-1,0.2) for s in fly[2,indx,:numtrials,'RightWing']]
    
    times = expl.ts(fly[2,indx,0,'LeftWing'][0],-1,0.2).times
    LRs = [expl.get_wbas(L,R) for L,R in zip(LeftWings,RightWings)]
    Lmean = expl.get_signal_mean([x[0] for x in LRs])
    Rmean = expl.get_signal_mean([x[1] for x in LRs])
    L_m_R_mean = expl.get_signal_mean([x[0]-x[1] for x in LRs])
    stim = expl.get_signal_mean([expl.expan_transform(expl.ts(s,-1,0.2)) for s in fly[2,indx,:numtrials,'Xpos']])
    plb.plot(times,stim,color = 'k')
    
    ######plot behavior#######
    plb.axes(ax1)#ax1 = plb.subplot(4,1,2,sharex = ax0)
    
    plb.plot(times,Lmean,color = (0.7,0.4,0.2))
    plb.plot(times,Rmean,color = (0.2,0.4,0.7))
    plb.plot(times,L_m_R_mean,color = plb.cm.Paired(0.3))
    
    ######plot raster########
    plb.axes(ax2)#ax2 = plb.subplot(4,1,3, sharex = ax0)
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
    filtered1 = [w for w in wvfm_list if w['id'] == 1]
    filtered2 = [w for w in wvfm_list if w['id'] == 0]
    #print len(filtered), 0.5*len(wvfm_list)
    if len(filtered1) > len(filtered2):
        filtered = filtered1
    else:
        filtered = filtered2
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
    plb.axes(ax3)#ax3 = plb.subplot(4,1,4, sharex = ax0)
    [plb.plot(np.array(trn),trn.annotations['phases']/np.pi,'o',color = 'k',alpha = 0.2) for trn in inc_trains]
    
    #stim axes
    ax0.set_xbound(-0.9,0.2)
    ax0.set_ybound(0,100)
    #behavior axes
    ax1.set_ybound(-6,10)
    #phase axes
    
    ax3.set_ybound(0,2)
    ax3.set_yticks([0,1,2])
    ax3.set_yticklabels([u'0',u'\u03C0',u'2\u03C0'])
    plb.savefig('cell_sum.pdf')
    plb.show()
    return idx,wvfm_list,inc_trains