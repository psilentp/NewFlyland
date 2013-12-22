import os
import neo
import scipy.io as sio
import quantities as pq
from pylab import *
import numpy as np
from scipy.signal import medfilt,find_peaks_cwt
import pylab as plb


def ts(sweep,start,stop):
    sta_index = argwhere(sweep.times >=start)[0]
    stp_index = argwhere(sweep.times >=stop)[0]
    return neo.AnalogSignal(sweep[sta_index:stp_index],
                            t_start = sweep.times[sta_index],
                            sampling_rate = sweep.sampling_rate)

def window_stdev(arr, window):
    from scipy.ndimage.filters import uniform_filter
    c1 = uniform_filter(arr, window*2, mode='constant', origin=-window)
    c2 = uniform_filter(arr*arr, window*2, mode='constant', origin=-window)
    return ((c2 - c1*c1)**.5)


def recondition_peaks(signal,pks):
    newpks = list()
    flips = list()
    for pk in pks:
        offset = np.argmax(signal[pk-10:pk+10])
        newpks.append(pk-10+offset)
        try:
            flip = argwhere( (diff(signal[pk-20:pk])>0) < 1 )[-1]+1
            flips.append(pk-20+int(flip))
        except IndexError:
            pass
    return newpks,flips

def _new_spikepool(cls,signal, t_stop, units=None, dtype=np.float,
                    copy=True, sampling_rate=None, t_start=0.0 * pq.s,
                    waveforms=None, left_sweep=None, name=None,
                    file_origin=None, description=None, annotations=None):
        
    return SpikePool(signal,
                t_stop = t_stop,
                units = units,
                sampling_rate = sampling_rate,
                waveforms = waveforms,
                left_sweep = left_sweep,
                t_start = t_start,
                **annotations)

class SpikePool(neo.SpikeTrain):
    """hold a Sub group of spikes for further processing - keep track of spike
    pool and spike pool indecies so it is easy to mix the results back in"""
    def __init__(self,*args,**argv):
        super(SpikePool,self).__init__(args,argv)
        self.wv_mtrx = np.vstack([np.array(wv) for wv in self.waveforms])
        self.spk_ind = np.arange(0,np.shape(self.wv_mtrx)[0])
            
    def copy_slice(self,sli,rezero = False):
        """copy a spiketrain with metadata also enable slicing"""
        st = self[sli]
        if rezero:
            shift = st.waveforms[0].times[0]
        else:
            shift = pq.Quantity(0,'s')
        wvfrms = [neo.AnalogSignal(np.array(wf),
                               units = wf.units,
                               sampling_rate = wf.sampling_rate,
                               name = wf.name,
                               t_start = wf.t_start - shift)
                               for wf in st.waveforms]
        #pk_ind = self.annotations['pk_ind'][sli]
        t_start = wvfrms[0].times[0]
        t_stop = wvfrms[-1].times[-1]
        return SpikePool(np.array(st)-float(shift),
                        units = st.units,
                        sampling_rate = st.sampling_rate,
                        waveforms = wvfrms,
                        left_sweep = st.left_sweep,
                        t_start = t_start,
                        t_stop = t_stop)
        
    def __reduce__(self):
    	return _new_spikepool, (self.__class__, np.array(self),
                                 self.t_stop, self.units, self.dtype, True,
                                 self.sampling_rate, self.t_start,
                                 self.waveforms, self.left_sweep,
                                 self.name, self.file_origin, self.description,
                                 self.annotations)
       
                                
def get_spike_pool(sweep,thresh = 10,wl=25,wr=20,filter_window = 35):
    from scipy.signal import medfilt
    detrend = np.array(sweep)-medfilt(sweep,filter_window)
    deltas = np.diff(np.array(detrend>thresh,dtype = 'float'))
    starts = np.argwhere(deltas>0.5)
    stops = np.argwhere(deltas<-0.5)
    if starts[0] > stops[0]:
        stops = stops[1:]
    if stops[-1] < starts[-1]:
        starts = starts[:-1]
    intervals = np.hstack((starts,stops))
    peaks = [np.argmax(sweep[sta:stp])+sta for sta,stp in intervals]
    waveforms = [sweep[pk-wl:pk+wr] for pk in peaks][2:-2]
    sweep.sampling_period.units = 's'
    pk_tms = sweep.times[array(peaks)][2:-2]
    spike_pool = SpikePool(pk_tms,
                                sweep.t_stop,
                                sampling_rate = sweep.sampling_rate,
                                waveforms = waveforms,
                                left_sweep = wl*sweep.sampling_period,
                                t_start = sweep.t_start,
                                pk_ind = peaks)
    return spike_pool


def get_wingbeats(wb_signal,filter_window = 151,thresh = 0.5):
    from scipy.signal import medfilt
    detrend = np.array(wb_signal)-medfilt(wb_signal,filter_window)
    deltas = np.diff(np.array(detrend>thresh,dtype = 'float'))
    starts = np.argwhere(deltas>0.5)
    stops = np.argwhere(deltas<-0.5)
    if starts[0] > stops[0]:
        stops = stops[1:]
    if stops[-1] < starts[-1]:
        starts = starts[:-1]
    intervals = np.hstack((starts,stops))
    peaks = [np.argmax(wb_signal[sta:stp])+sta for sta,stp in intervals]
    return peaks
    
def get_signal_mean(signal_list):
    """calculate the average signal from a list of signals"""
    #print signal_list
    mean_signal = reduce(lambda x,y:x+y,signal_list)/len(signal_list)
    return mean_signal

def get_signal_stderr(signal_list,signal_mean = None):
    """return the standard error of a group of signals"""
    if signal_mean is None:
        signal_mean = get_signal_mean(signal_list)
    mean_sq_err = get_signal_mean(map(lambda x: (x-signal_mean)**2,signal_list))
    #print sqrt(mean_sq_err)/sqrt(len(signal_list))
    rsig = sqrt(mean_sq_err)/sqrt(len(signal_list))
    rsig.sampling_period = signal_list[0].sampling_period
    rsig.t_start = signal_list[0].t_start
    return rsig

def get_low_filter(signal,Hz):
    #return a low-pass-filterd version of the time series
    from scipy.signal import filtfilt
    filter_order = 3
    sampfreq = signal.sampling_rate
    filtfreq = pq.Quantity(Hz,'Hz')
    passband_param = float((filtfreq/sampfreq).simplified)
    from scipy.signal import butter
    [b,a]=butter(filter_order,passband_param)
    #ob = cp.copy(signal)
    filt_array = filtfilt(b,a,signal)
    #ob.y = quan.Quantity(filtfilt(b,a,self.y),self.yunits)
    return neo.AnalogSignal(filt_array,
                            units = signal.units,
                            sampling_rate = signal.sampling_rate,
                            t_start = signal.t_start)
        
def calculate_groupwise_means(flylist = []):
    import cPickle as cpkl
    group_matrix = dict()
    average_matrix = dict()
    stde_matrix = dict()
    #flylist = [1,2,3,4,5,6,7,9,10,11,12]
    for findex in range(5):
            for pindex in[4,10]:
                group_matrix['Xpos',findex+1,pindex+1] = list()
                group_matrix['L_m_R',findex+1,pindex+1] = list()
                average_matrix['Xpos',findex+1,pindex+1] = list()
                average_matrix['L_m_R',findex+1,pindex+1] = list()  
                stde_matrix['Xpos',findex+1,pindex+1] = list()
                stde_matrix['L_m_R',findex+1,pindex+1] = list()
    #load the group matrix with fly data
    for flynum in flylist:
        fi = open('fly%s.cpkl'%(flynum),'rb')
        fly_data = cpkl.load(fi)
        print("loading fly%s"%(flynum))
        for findex in range(5):
            for pindex in[4,10]:
                group_matrix['Xpos',findex+1,pindex+1].append(fly_data['Xpos',findex+1,pindex+1])
                group_matrix['L_m_R',findex+1,pindex+1].append(fly_data['L_m_R',findex+1,pindex+1])
    for findex in range(5):
        for pindex in [4,10]:
            print('averaging cell %s %s'%(findex,pindex))
            average_matrix['Xpos',findex+1,pindex+1] = get_signal_mean(group_matrix['Xpos',findex+1,pindex+1])
            average_matrix['L_m_R',findex+1,pindex+1] = get_signal_mean(group_matrix['L_m_R',findex+1,pindex+1])
            stde_matrix['Xpos',findex+1,pindex+1] = get_signal_stderr(group_matrix['Xpos',findex+1,pindex+1],signal_mean = average_matrix['Xpos',findex+1,pindex+1])
            stde_matrix['L_m_R',findex+1,pindex+1] = get_signal_stderr(group_matrix['L_m_R',findex+1,pindex+1],signal_mean = average_matrix['L_m_R',findex+1,pindex+1])
    return average_matrix,stde_matrix,group_matrix


def get_spks_in_wbs(wbt,sp):
    wb_window = 10000
    left_ind = 0
    spk_wbind = zeros_like(array(sp))
    wbt_a = array(wbt)
    sp_a = array(sp)
    wb_len = shape(wbt_a)[0]
    sp_len = shape(array(sp))[0]
    for i in range(sp_len):
        right_ind = left_ind+wb_window
        if right_ind >= wb_len:
            right_ind = wb_len-1
        x = sp_a[i] - wbt_a[left_ind:right_ind-1,newaxis].T > 0
        x2 = sp_a[i] - wbt_a[left_ind+1:right_ind,newaxis].T > 0
        try:
            wb_ind = argwhere(~x2.T & x.T)[0,0] + left_ind
        except ValueError:
            print sp_len-1
            print right_ind
        spk_wbind[i] =  wb_ind
        if wb_ind > left_ind:
            left_ind = wb_ind-1
    return spk_wbind
    
def get_trials(fly,function_index,position_index,sli,sigkey):
    import defaults as df
    retlist = list()
    selected_trials_indices = [rps[0] for rps in fly.trial_matrix if rps[1] == function_index and rps[2] == position_index][sli]
    for x in selected_trials_indices:
        #first get the section presumed to include t-collision
        start_ind = fly.trial_start_indicies[x]-fly.left_window
        end_ind = fly.trial_start_indicies[x]+fly.right_window
        #if self.get_wbf(start_ind,end_ind) > self.min_wbf:
        #shift the start and end ind to correct for the actual collision time.
        icol = where(fly.signals['Xpos'][fly.trial_start_indicies[x]+5000:end_ind] >= df.ol_volts_per_deg*89.5)[0][0]
        offset = icol + 5000 - floor(int(fly.olepoch/fly.dt)) - 1
        start_ind += offset
        end_ind += offset
        tsig = fly.signals['Xpos'].times
        start_time = tsig[start_ind]
        end_time = tsig[end_ind]
        if sigkey in fly.processed_signals.keys():
            if sigkey == 'spike_pool':
                sp = fly.processed_signals['spike_pool']
                sp_sind = argwhere(sp>start_time)[0]
                sp_eind = argwhere(sp>end_time)[0] 
                ### accessory signals
                mask = fly.processed_signals['spike_mask']
                sp_phases = fly.processed_signals['spk_phases']
                sp_wbs = fly.processed_signals['spk_wbs']
                ####cut up accessory signals
                print sp_sind,sp_eind
                cut_sp = sp.copy_slice(slice(sp_sind,sp_eind),rezero = False)
                cut_mask = mask[sp_sind:sp_eind]
                cut_phases = sp_phases[sp_sind:sp_eind]
                cut_wbs = sp_wbs[sp_sind:sp_eind]
                ####send them back in annotations
                cut_sp.annotations['spike_mask'] = cut_mask
                cut_sp.annotations['spk_phases'] = cut_phases
                cut_sp.annotations['spk_wbs'] = cut_wbs
                retlist.append(cut_sp)
            else:
                retlist.append(fly.processed_signals[sigkey][start_ind:end_ind])
        if sigkey in fly.signals.keys():
            sig = fly.signals[sigkey][start_ind:end_ind]
            retlist.append(sig)
            
    return retlist
    
def calc_spike_phases(spike_pool,phases,times):
    phase_idxs = np.zeros_like(np.array(spike_pool),dtype = int)
    start_idx = 0
    search_range = 1000
    max_len = len(times)
    for i,spk in enumerate(spike_pool):
        try:
            phase_idxs[i] = np.argwhere(np.diff(spk>times[start_idx:search_range]))[0][0]
        except IndexError:
            search_range = max_len-1
            phase_idxs[i] = np.argwhere(np.diff(spk>times[start_idx:search_range]))[0][0]
        phase_idxs[i] += start_idx
        start_idx = phase_idxs[i]
        search_range = 1000+start_idx
        if search_range > max_len-1:
            search_range = max_len-1
    return phases[phase_idxs]

def calc_spike_wbs(spike_pool,flips,times):
    flip_tms = times[flips]
    spike_wbs = np.zeros_like(np.array(spike_pool),dtype = int)
    start_search = 0
    end_search = 1000
    search_len = len(spike_pool)
    for i,spike in enumerate(spike_pool):
        try:
            spike_wbs[i] = argwhere(spike<flip_tms[start_search:end_search])[0]+start_search
        except IndexError:
            end_search = search_len
            try:
                spike_wbs[i] = argwhere(spike<flip_tms[start_search:end_search])[0]+start_search
            except IndexError:
                pass
                #print spike
                #print flip_tms[start_search:end_search]
                #print argwhere(spike<flip_tms[start_search:end_search])
        start_search = spike_wbs[i]
        end_search = start_search + 1000
        if end_search > search_len:
            end_search = search_len
    return spike_wbs

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