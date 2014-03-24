#! /Users/psilentp/Library/Enthought/Canopy_64bit/User/bin/Python

import sys

flynum = int(sys.argv[1])

###################
###################
from pylab import *
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import sys
from sklearn.cluster import KMeans

analysis_lib = '/Users/psilentp/Documents/Projects/NewFlyland/Analysis/'
sys.path.append(analysis_lib)

import flyphys as fph
import defaults
import sorters as srtr
import sorter_view as sview
import signal_tools as stools
dataroot = '/Users/psilentp/Dropbox/Data/LeftRight/'

###################
###################
ex_map = {
         27:{'ipsi_pole':5 ,
             'contra_pole':11,
             'ipsi_key':'LeftAmp',
             'contra_key':'RightAmp'},
         28:{'ipsi_pole':5 ,
             'contra_pole':11,
             'ipsi_key':'LeftAmp',
             'contra_key':'RightAmp'},
         29:{'ipsi_pole':5 ,
             'contra_pole':11,
             'ipsi_key':'LeftAmp',
             'contra_key':'RightAmp'},
         32:{'ipsi_pole':5 ,
             'contra_pole':11,
             'ipsi_key':'LeftAmp',
             'contra_key':'RightAmp'},
         33:{'ipsi_pole':5 ,
             'contra_pole':11,
             'ipsi_key':'LeftAmp',
             'contra_key':'RightAmp'},
         35:{'ipsi_pole':5 ,
             'contra_pole':11,
             'ipsi_key':'LeftAmp',
             'contra_key':'RightAmp'},
         36:{'ipsi_pole':5 ,
             'contra_pole':11,
             'ipsi_key':'LeftAmp',
             'contra_key':'RightAmp'},
         37:{'ipsi_pole':5 ,
             'contra_pole':11,
             'ipsi_key':'LeftAmp',
             'contra_key':'RightAmp'},
         38:{'ipsi_pole':5 ,
             'contra_pole':11,
             'ipsi_key':'LeftAmp',
             'contra_key':'RightAmp'},
         40:{'ipsi_pole':5 ,
             'contra_pole':11,
             'ipsi_key':'LeftAmp',
             'contra_key':'RightAmp'},
         41:{'ipsi_pole':5 ,
             'contra_pole':11,
             'ipsi_key':'LeftAmp',
             'contra_key':'RightAmp'},
         42:{'ipsi_pole':5 ,
             'contra_pole':11,
             'ipsi_key':'LeftAmp',
             'contra_key':'RightAmp'},
         43:{'ipsi_pole':5 ,
             'contra_pole':11,
             'ipsi_key':'LeftAmp',
             'contra_key':'RightAmp'},
         44:{'ipsi_pole':5 ,
             'contra_pole':11,
             'ipsi_key':'LeftAmp',
             'contra_key':'RightAmp'},
         45:{'ipsi_pole':5 ,
             'contra_pole':11,
             'ipsi_key':'LeftAmp',
             'contra_key':'RightAmp'},
         47:{'ipsi_pole':5 ,
             'contra_pole':11,
             'ipsi_key':'LeftAmp',
             'contra_key':'RightAmp'},
         49:{'ipsi_pole':11 ,
             'contra_pole':5,
             'ipsi_key':'RightAmp',
             'contra_key':'LeftAmp'}
        }
        

###################
###################

def extract_ol_spike_rasters(fly,
                             ypos,
                             ipsi_key = 'LeftAmp',
                             contra_key = 'RightAmp'):
    trials = list()
    #*********
    sig_slice = slice(0,10)
    
    x_pos = stools.get_trials(fly,2,ypos,sig_slice,'Xpos')
    e_sweeps = stools.get_trials(fly,2,ypos,sig_slice,'AMsysCh1')
    spks = stools.get_trials(fly,2,ypos,sig_slice,'spike_pool')
    spk_phases = [x.annotations['spk_phases'] for x in spks]
    spk_masks = [x.annotations['spike_mask'] for x in spks]
    spk_wb_idxs = [x.annotations['spk_wbs'] for x in spks]
    flight_masks = stools.get_trials(fly,2,ypos,sig_slice,'flight_mask')
    
    full_flights = [x for x in range(len(flight_masks)) if not(np.mean(flight_masks[x]))]
    
    contra_amp = stools.get_trials(fly,2,ypos,sig_slice,ipsi_key)
    ipsi_amp = stools.get_trials(fly,2,ypos,sig_slice,contra_key)
    print full_flights
    ##filter out the non flights
    for trl_idx in full_flights:
        swp_start = e_sweeps[trl_idx].times[0]
        pre_col_epoch = -1*(int(fly.clepoch/fly.dt)+int(fly.olepoch/fly.dt)+1)*fly.dt
        swp_times = e_sweeps[trl_idx].times - swp_start + pre_col_epoch
        f_spk_times = list()
        f_spk_phases = list()
        f_spk_wbs = list()
        ###filter out the masked spikes
        for spt,phs,msk,wbs in zip(spks[trl_idx],spk_phases[trl_idx],spk_masks[trl_idx],spk_wb_idxs[trl_idx]):
            if msk:
                f_spk_times.append(spt-swp_start+pre_col_epoch)
                f_spk_phases.append(phs)
                f_spk_wbs.append(wbs)
        #get the wingbeat amp data for each spike
        f_spk_flips = fly.processed_signals['wb_flips'][np.array(f_spk_wbs,dtype=int)+1]
        f_spk_ipsi_amp = fly.processed_signals[ipsi_key][f_spk_flips]
        f_spk_contra_amp = fly.processed_signals[contra_key][f_spk_flips]
        ###add a down-sampled time-array for each trial left/right wba and x-pos.
        down_x_pos = x_pos[trl_idx][::10]
        down_ipsi_amp = ipsi_amp[trl_idx][::10]
        down_contra_amp = contra_amp[trl_idx][::10]
        down_times = swp_times[::10]
        trials.append({'f_spk_times':np.array(f_spk_times),
                       'f_spk_phases':np.array(f_spk_phases),
                       'f_spk_wbs':np.array(f_spk_wbs),
                       'f_spk_flips':f_spk_flips,
                       'f_spk_ipsi_amp':f_spk_ipsi_amp,
                       'f_spk_contra_amp':f_spk_contra_amp,
                       'down_x_pos':down_x_pos,
                       'down_ipsi_amp':down_ipsi_amp,
                       'down_contra_amp':down_contra_amp,
                       'down_times':down_times})
    return trials
    
def phase_raster(event_times_list,phase_list):
    ax = gca()
    for ith,trial in enumerate(zip(event_times_list,phase_list)):
        colors = cm.jet(trial[1]/(2*np.pi))
        vlines(trial[0],ith + 0.5, ith+ 1.5,color = colors)
    return ax
    
def time_bin_phases(spk_rasters,
                      start = -0.3,
                      stop = 0.1,
                      interval = 0.01):
    cat_times = concatenate([r['f_spk_times'] for r in spk_rasters])
    cat_phases = concatenate([r['f_spk_phases'] for r in spk_rasters])
    cat_ipsi_amp = concatenate([r['f_spk_ipsi_amp'] for r in spk_rasters])
    cat_contra_amp = concatenate([r['f_spk_contra_amp'] for r in spk_rasters])

    bins = arange(start,stop,interval)
    med_phase = list()
    top_phase = list()
    bot_phase = list()

    for b in bins:
        subsample = argwhere((cat_times>b) & (cat_times < b+interval))
        med_phase.append(median(cat_phases[subsample]))
        top_phase.append(percentile(cat_phases[subsample],90))
        bot_phase.append(percentile(cat_phases[subsample],10))
    
    return bins+(interval/2.0),med_phase,top_phase,bot_phase
    

    
    
###################
#load the fly
###################
print("loading Fly %s"%(flynum))
fly = fph.get_fly_in_rootdir(dataroot,flynum,0,rec_location = 'B1')
fly_cntrlr = fph.FlyController(fly)

###################
#get exp settings
###################

ipsi_pole = ex_map[flynum]['ipsi_pole'] 
contra_pole = ex_map[flynum]['contra_pole'] 
ipsi_key = ex_map[flynum]['ipsi_key']
contra_key = ex_map[flynum]['contra_key']

###################
#fix the wb data
###################

print("correcting for short wingbeats")


wb_flips = fly.processed_signals['wb_flips']
fix_mask = hstack((diff(wb_flips)<60,array(True)))
fixed_wb_flips = delete(wb_flips,argwhere(fix_mask))

wb_peaks = fly.processed_signals['wb_peaks']
fix_mask = hstack((diff(wb_peaks)<60,array(True)))
fixed_wb_peaks = delete(wb_peaks,argwhere(fix_mask))

times = fly.signals['LeftWing'].times
sp = fly.processed_signals['spike_pool']
fly.processed_signals['spk_wbs'] = stools.calc_spike_wbs(sp,fixed_wb_flips,times)

L_h = fly.signals['LeftWing']
R_h = fly.signals['RightWing']

L_a = np.zeros_like(L_h)
R_a = np.zeros_like(R_h)
p0 = fixed_wb_peaks[0]
for pk in fixed_wb_peaks[1:]:
    L_a[p0:pk] = L_h[p0]
    R_a[p0:pk] = R_h[p0]
    p0 = pk
L_a[0:fixed_wb_peaks[0]] = L_h[0]
R_a[0:fixed_wb_peaks[0]] = R_h[0]
L_a[fixed_wb_peaks[-1]:] = L_h[fixed_wb_peaks[-1]]
R_a[fixed_wb_peaks[-1]:] = R_h[fixed_wb_peaks[-1]]

fly.processed_signals['wb_flips'] = fixed_wb_flips
fly.processed_signals['wb_peaks'] = fixed_wb_peaks
fly.processed_signals['LeftAmp'] = L_a
fly.processed_signals['RightAmp'] = R_a

###################
#Get the phase rasters
###################
print("loading phase rasters")
spk_rasters_i = extract_ol_spike_rasters(fly,ipsi_pole,ipsi_key,contra_key)
spk_rasters_c = extract_ol_spike_rasters(fly,contra_pole,ipsi_key,contra_key)


###################
#plot summary fig
###################
print("plotting summary fig")
sum_fig = figure(figsize = (8,12))

dtimes = spk_rasters_i[0]['down_times']
ax00 = subplot(4,2,1)
plot(dtimes,mean(vstack([array(rstr['down_x_pos']) for rstr in spk_rasters_i]),axis = 0))
ax01 = subplot(4,2,3,sharex = ax00)
plot(dtimes,mean(vstack([array(rstr['down_contra_amp']) for rstr in spk_rasters_i]),axis = 0))
plot(dtimes,mean(vstack([array(rstr['down_ipsi_amp']) for rstr in spk_rasters_i]),axis = 0))
ax02 = subplot(4,2,5,sharex = ax00)
phase_raster([r['f_spk_times'] for r in spk_rasters_i],[r['f_spk_phases'] for r in spk_rasters_i])
ax03 = subplot(4,2,7,sharex = ax00)
for rstr in spk_rasters_i:
    plot(rstr['f_spk_times'],rstr['f_spk_phases'],'o',alpha = 0.2,color = 'k')
bins,med_phase,top_phase,bot_phase = time_bin_phases(spk_rasters_i)
plot(bins,med_phase,'o-',color ='r')
plot(bins,top_phase,color ='r')
plot(bins,bot_phase,color ='r')



ax10 = subplot(4,2,2)
plot(dtimes,mean(vstack([array(rstr['down_x_pos']) for rstr in spk_rasters_c]),axis = 0))
ax11 = subplot(4,2,4,sharex = ax10)
plot(dtimes,mean(vstack([array(rstr['down_contra_amp']) for rstr in spk_rasters_c]),axis = 0))
plot(dtimes,mean(vstack([array(rstr['down_ipsi_amp']) for rstr in spk_rasters_c]),axis = 0))
ax12 = subplot(4,2,6,sharex = ax10)
phase_raster([r['f_spk_times'] for r in spk_rasters_c],[r['f_spk_phases'] for r in spk_rasters_c])
ax13 = subplot(4,2,8,sharex = ax10)
for rstr in spk_rasters_c:
    plot(rstr['f_spk_times'],rstr['f_spk_phases'],'o',alpha = 0.2,color = 'k')
bins,med_phase,top_phase,bot_phase = time_bin_phases(spk_rasters_c)
plot(bins,med_phase,'o-',color = 'r')
plot(bins,top_phase,color ='r')
plot(bins,bot_phase,color ='r')

ax00.set_xbound(-0.3,0.1)
ax10.set_xbound(-0.3,0.1)
ax00.set_ybound(0,3)
ax10.set_ybound(0,3)
ax01.set_ybound(0,10)
ax11.set_ybound(0,10)

ax03.set_ybound(0,2*pi)
ax13.set_ybound(0,2*pi)



###################
#plot scatter_data
###################

scat_fig = figure(figsize=(4,8))
cat_times = concatenate([r['f_spk_times'] for r in spk_rasters_i])
cat_phases = concatenate([r['f_spk_phases'] for r in spk_rasters_i])
cat_ipsi_amp = concatenate([r['f_spk_ipsi_amp'] for r in spk_rasters_i])
cat_contra_amp = concatenate([r['f_spk_contra_amp'] for r in spk_rasters_i])


ax0 = subplot(2,1,1)
subsample = argwhere((cat_times>-0.05) & (cat_times < 0.05))
scatter(cat_ipsi_amp,cat_phases,c = 'k',alpha = 0.1)
scatter(cat_ipsi_amp[subsample],cat_phases[subsample],c = 'r')

ax1 = subplot(2,1,2)
subsample = argwhere((cat_times>-0.3) & (cat_times < 0.1))
colors = squeeze(cm.jet(cat_phases[subsample]/(2*pi)))
scatter(cat_times[subsample],cat_ipsi_amp[subsample],c = colors)
ax1.set_xbound(-.3,0.1)

sum_fig.savefig('Summary_Fly%s'%(flynum))
scat_fig.savefig('Scatter_Fly%s'%(flynum))

import cPickle
f = open('summmary_data_Fly%s.cpkl'%(flynum),'wb')
cPickle.dump([spk_rasters_i,spk_rasters_c],f)
f.close()