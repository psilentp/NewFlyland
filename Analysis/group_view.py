def plot_sweeps(fly,
                electrode,
                spks,
                spk_mask,
                spk_phases,
                ipsi_wing,
                ipsi_amp,
                contra_wing,
                contra_amp,
                wb_phases,
                x_pos,
                trl_idx = 0):
    swp_start = ipsi_amp[trl_idx].times[0]
    t_start = -1*(int(fly.clepoch/fly.dt)+int(fly.olepoch/fly.dt)+1)*fly.dt#-2*(self.epoch)
    swp_times = ipsi_amp[trl_idx].times - swp_start + t_start

    ax = subplot(4,1,1)
    plot(swp_times,x_pos[trl_idx],color = 'k')
    
    
    subplot(4,1,2,sharex = ax)
    plot(swp_times,ipsi_wing[trl_idx],color = 'b',alpha = 0.5)
    subplot(4,1,2,sharex = ax)
    plot(swp_times,contra_wing[trl_idx],color = 'g',alpha = 0.5)
    subplot(4,1,2,sharex = ax)
    plot(swp_times,ipsi_amp[trl_idx],color = 'b')
    subplot(4,1,2,sharex = ax)
    plot(swp_times,contra_amp[trl_idx],color = 'g')
    
    #spk_lins = [plot(wf.times-swp_start+t_start,wf,color = cm.jet(float(phs/(2*pi)))) for wf,phs in zip(spks[trl_idx].waveforms,spks[trl_idx].annotations['spk_phases'])]

    
    subplot(4,1,4,sharex = ax)
    plot(swp_times,electrode[trl_idx],color = 'k')
    f_stimes = list()
    f_sphase = list()
    f_colors = list()
    for spt,wf,phs,msk in zip(spks[trl_idx],spks[trl_idx].waveforms,spk_phases[trl_idx],spk_mask[trl_idx]):
        if msk:
            color = cm.jet(float(phs/(2*pi)))
            plot(wf.times-swp_start+t_start, wf, color = color,lw = 5,alpha = 0.5,)
            #print shape(spt)
            f_stimes.append(spt-swp_start+t_start)
            f_sphase.append(phs)
            f_colors.append(color)
    
    subplot(4,1,3,sharex = ax)
    plot(swp_times,wb_phases[trl_idx],color = 'k')
    scatter(array(f_stimes),array(f_sphase),c = f_colors,s = 40)
    ax.set_xbound(-0.08,0.02)
    ax.set_ybound(0,3)
    
def extract_cl_spike_rasters(fly,
                             start_time,
                             end_time,
                             e_sweeps,
                             spks,
                             spk_phases,
                             spk_masks,
                             spk_wb_idxs,
                             full_flights):
    trials = list()
    for trl_idx in full_flights:
        swp_start = e_sweeps[trl_idx].times[0]
        t_start = -1*(int(fly.clepoch/fly.dt)+int(fly.olepoch/fly.dt)+1)*fly.dt#-2*(self.epoch)
        swp_times = e_sweeps[trl_idx].times - swp_start + t_start
        f_spk_times = list()
        f_spk_phases = list()
        f_spk_wbs = list()
        for spt,phs,msk,wbs in zip(spks[trl_idx],spk_phases[trl_idx],spk_masks[trl_idx],spk_wb_idxs[trl_idx]):
            if msk:
                f_spk_times.append(spt-swp_start+t_start)
                f_spk_phases.append(phs)
                f_spk_wbs.append(wbs)
        trials.append({'f_spk_times':np.array(f_spk_times),
                       'f_spk_phases':np.array(f_spk_phases),
                       'f_spk_wbs':np.array(f_spk_wbs)})
    return trials