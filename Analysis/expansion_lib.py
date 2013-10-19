#script to parse the data for a single fly - load % cut up the .abf and matlab format and save the mean traces in a python friendly format.
import os
import neo
import scipy.io as sio
import quantities as pq
from pylab import *
import numpy as np
from scipy.signal import medfilt,find_peaks_cwt
import pylab as plb

#fly_number = 1
#rep_ind = 0
dataroot = '/Users/psilentp/Dropbox/Data/FirstEphys'

azmuth_lookup = {5:90,11:-90}
L_V_lookup = {1:5,2:20,3:50,4:100,5:200}


##transform to convert expansion x signal into degrees
pattern_xnum = 768
deg_per_xstp = 360./pattern_xnum
ol_volts_per_deg = 10.0/(pattern_xnum*deg_per_xstp)
cl_volts_per_rad = 5/(np.pi)
expan_transform = lambda x:x/ol_volts_per_deg
fix_transform = lambda x:x/cl_volts_per_rad

class FlyRecord(object):
    def __init__(self,fly_number,rep_ind,dataroot):
        self.fly_number = fly_number
        self.datadir = dataroot + '/Fly%s'%(fly_number)
        self.rep_ind = rep_ind
        self.load_data()
        #window of when the trial begins and ends
        #self.epoch = pq.Quantity(5,'s')
        self.clepoch = pq.Quantity(5,'s')
        self.olepoch = pq.Quantity(1.2,'s')
        self.tail = pq.Quantity(.4,'s')
        self.left_window = int(self.clepoch/self.dt)
        self.right_window = int(self.olepoch/self.dt + self.tail/self.dt)
        self.min_wbf = pq.Quantity(150.0,'Hz')
    
    def __getitem__(self,k):
        function_index = k[0]
        position_index = k[1]
        sli = k[2]
        sigkey = k[3]
        #try:
        #    sli = k[3]
        #except IndexError:
        #    pass
        retlist = list()
        selected_trials_indices = [rps[0] for rps in self.trial_matrix if rps[1] == function_index and rps[2] == position_index][sli]
        if not(type(selected_trials_indices) is list):
            selected_trials_indices = [selected_trials_indices]
        for x in selected_trials_indices:
            #first get the section presumed to include t-collision
            start_ind = self.trial_start_indicies[x]-self.left_window
            end_ind = self.trial_start_indicies[x]+self.right_window
            #if self.get_wbf(start_ind,end_ind) > self.min_wbf:
            #shift the start and end ind to correct for the actual collision time.
            icol = where(self.signals['Xpos'][self.trial_start_indicies[x]+5000:end_ind] >= ol_volts_per_deg*89.5)[0][0]
            offset = icol + 5000 - floor(int(self.olepoch/self.dt)) - 1
            start_ind += offset
            end_ind += offset
            retlist.append(self.signals[sigkey][start_ind:end_ind])
        for sig in retlist:
            #set the start time correcting for rounding errors
            sig.t_start = -1*(int(self.clepoch/self.dt)+int(self.olepoch/self.dt)+1)*self.dt#-2*(self.epoch)
        #print [s.t_start for s in retlist]
        return retlist
    
    def extract_trial_spikes(self,function_index, position_index,trial_num,start,stop):
        from scipy.signal import hilbert
        L_h = self[function_index,position_index,trial_num,'LeftWing'][0]
        R_h = self[function_index,position_index,trial_num,'RightWing'][0]
        #start_ind = int(start/L_h.sampling_period)
        #stop_ind = int(stop/L_h.sampling_period)
        start_ind = np.argwhere(L_h.times>=start)[0]
        stop_ind = np.argwhere(L_h.times>=stop)[0]
        #phases = np.angle(hilbert(get_low_filter(L_h+R_h,500)))
        phases = get_phase_trace(L_h[start_ind:stop_ind],R_h[start_ind:stop_ind])
        spk_train = get_spiketrain(self[function_index,position_index,trial_num,'AMsysCh1'][0][start_ind:stop_ind])
        spk_ind = spk_train.annotations['pk_ind']
        spk_train.annotations['phases'] = phases[spk_ind]
        return spk_train
        
    def get_wbf(self,start_ind,end_ind):
        duration = (end_ind-start_ind)*self.signals['Sync'].times[1]
        wingbeats = len(where(diff(where(self.signals['Sync'][start_ind:end_ind]<1,1,0))>0.5)[0])
        freq =  wingbeats/duration
        return freq
    
    def load_data(self):
        self.datafiles = os.listdir(self.datadir)
        self.abf_files = [f for f in self.datafiles if '.abf' in f]
        self.mat_files = [f for f in self.datafiles if '.mat' in f]
        importer = neo.io.AxonIO(filename = self.datadir + '/' + self.abf_files[self.rep_ind])
        block = importer.read_block()
        hdr = importer.read_header()
        #holds the asignals
        seg = block.segments[0]
        #get and sort the trial presentation record
        trial_matrix = sio.loadmat(self.datadir + '/' + self.mat_files[self.rep_ind])['datarecord'][0]
        #now add an index to the first column.
        self.trial_matrix = np.array([[i,x[0][0][0],x[1][0][0]] for i,x in enumerate(trial_matrix)])
        #sort the data: first by presentation order, then by function type, then by position.
        self.sorted_trial_indices = lexsort((self.trial_matrix[:,0],self.trial_matrix[:,1],self.trial_matrix[:,2]))
        #make a lookup dictionary for the signal indicies
        self.signals = dict()
        for index,sig_name in enumerate(h['ADCChNames'] for h in hdr['listADCInfo']):
            self.signals[sig_name] = seg.analogsignals[index]
        #pull out the trial start indicies
        self.dt = self.signals['Sync'].times[1]
        self.trial_start_indicies = where(diff(where(self.signals['Ypos'] <1,1,0))>0)[0]
        
    def plot_trialtype(self, signal_key, function_index, position_index,plot_range,transform = None):
        siglist = self[function_index,position_index,:,signal_key]
        #print siglist[0].times
        #siglist = self.__getitem__((signal_key,function_index,position_index))
        st = np.argwhere(siglist[0].times<plot_range[0])[-1]
        en = np.argwhere(siglist[0].times<plot_range[1])[-1]
        if signal_key == 'AMsysCh1':
            downsamp = 1
        else:
            downsamp = 100
        if transform:
            siglist = [transform(x[st:en]) for x in siglist]
        else:
            siglist = [x[st:en] for x in siglist]
            
        for sig in siglist:
            plot(sig.times[::downsamp],sig[::downsamp],color = 'k', alpha = 0.2)
        if not (signal_key is 'AMsysCh1'):
            ave = reduce(lambda x,y:x+y,siglist)/len(siglist)
            plot(ave.times[::downsamp],ave[::downsamp],color = 'r')
        else:
            new_siglist = self[function_index,position_index,:,'L_m_R']
            #new_siglist = self.__getitem__(('L_m_R',function_index,position_index))
            ave = reduce(lambda x,y:x+y,new_siglist)/len(new_siglist)
            plot(ave.times[::downsamp],ave[::downsamp]*2,color = 'r')
    
    def plot_closedloop(self,function_index,position_index,plot_range):
        #siglist = self.__getitem__(('Xpos',function_index,position_index))
        siglist = self[function_index,position_index,:,'Xpos']
        st = np.argwhere(siglist[0].times<plot_range[0])[-1]
        en = np.argwhere(siglist[0].times<plot_range[1])[-1]
        data = np.concatenate([fix_transform(x[st:en]) for x in siglist])
        hist(data,bins= 369,alpha = 0.5,normed = True)
   
        
    def calculate_average(self, signal_key, function_index, position_index,ave_range,transform = None):
        #siglist = self.__getitem__((signal_key,function_index,position_index))
        siglist = self[function_index,position_index,:,signal_key]
        st = np.argwhere(siglist[0].times<ave_range[0])[-1]
        en = np.argwhere(siglist[0].times<ave_range[1])[-1]
        if transform:
            siglist = [transform(x[st:en]) for x in siglist]
        else:
            siglist = [x[st:en] for x in siglist]
        ave = reduce(lambda x,y:x+y,siglist)/len(siglist)  
        return ave

    def plot_closed_loop_summary(self):
        flynum = self.fly_number
        include_start = pq.Quantity(-7,'s')
        include_end = pq.Quantity(-5.5,'s')
        include_window = (include_start,include_end)
    
        fig = figure(figsize = (8,11))
        ax1 = subplot(6,2,1)
        ax2 = subplot(6,2,2,sharex = ax1)
        count = 3
        for findex in range(5):
            for pindex in [4,10]:
                if pindex == 4:
                    ax1 = subplot(6,2,1)
                else:
                    ax2 = subplot(6,2,2)
                
                ax3 = subplot(6,2,count,polar = True)
                self.plot_closedloop(findex+1,pindex+1,include_window)
                [x.set_visible(False) for x in ax3.get_xticklabels()]
                [x.set_visible(False) for x in ax3.get_yticklabels()]
                count += 1

        ax1.set_title('pole =  90')
        ax2.set_title('pole = -90')
        return fig
    
    def get_closed_loop_histogram(self,savefig = False):
        flynum = self.fly_number
        include_start = pq.Quantity(-7,'s')
        include_end = pq.Quantity(-5.0,'s')
        #include_window = (include_start,incl  de_end)
        #sig_zero = self['Xpos',1,5][0]
        sig_zero = self[1,5,0,'Xpos'][0]
        st = np.argwhere(sig_zero.times<include_start)[-1]
        en = np.argwhere(sig_zero.times<include_end)[-1]
        #np.concatenate([fix_transform(x[st:en]) for x in siglist]
        siglist = list()
        for findex in range(1,6):
            for pindex in [5,11]:
                siglist.extend([fix_transform(x[st:en]) for x in self[findex,pindex,:,'Xpos']])
        return np.histogram(np.concatenate(siglist),bins = 94,density = True)
    
    def plot_ephys_sweep(self,findex,pindex,sweepnum):
        times = self[findex,pindex:,'AMsysCh1'][sweepnum].times[-12000:-6000]
        fig = figure(figsize=(6,12))
        ax1 = subplot(3,1,1)
        plot(times,expan_transform(self[findex,pindex,:,'Xpos'][sweepnum][-12000:-6000]))
        ax2 = subplot(3,1,2,sharex = ax1)
        plot(times,self[findex,pindex,:,'AMsysCh1'][sweepnum][-12000:-6000])
        ax3 = subplot(3,1,3,sharex = ax1)
        plot(times,self[findex,pindex,:,'LeftWing'][sweepnum][-12000:-6000])
        plot(times,self[findex,pindex,:,'RightWing'][sweepnum][-12000:-6000])
        return fig
    
    def plot_ephys_summary(self):
        flynum = self.fly_number
        plot_start = pq.Quantity(-0.5,'s')
        plot_end = pq.Quantity(0.2,'s')
        plot_window = (plot_start,plot_end)
        fig = figure(figsize = (10,6))
        ax1 = subplot(6,2,1)
        ax2 = subplot(6,2,2,sharex = ax1)
     
        count = 3
        for findex in range(5):
            for pindex in [4,10]:
                if pindex == 4:
                    ax1 = subplot(6,2,1)
                else:
                    ax2 = subplot(6,2,2)
                self.plot_trialtype('Xpos',findex+1,pindex+1,(plot_window[0],plot_window[1]),transform = expan_transform)
                ax3 = subplot(6,2,count,sharex = ax1)
                self.plot_trialtype('AMsysCh1',findex+1,pindex+1,(plot_window[0],plot_window[1]))
                if pindex == 4:
                    ax3.set_ylabel('l/v =%s ms'%(L_V_lookup[findex+1]))
                ax3.set_ybound(-25,25)
                ax1.set_xbound(float(plot_start),float(plot_end))
                ax1.set_ybound(0,95)
                ax2.set_xbound(float(plot_start),float(plot_end))
                ax2.set_ybound(0,95)
                count +=1

        ax1.set_title('pole =  90')
        ax2.set_title('pole = -90')
        suptitle('Fly%s'%(flynum))
        return fig
    
    def plot_fly_summary(self):
        flynum = self.fly_number

        plot_start = pq.Quantity(-5,'s')
        plot_end = pq.Quantity(0.5,'s')
    
        plot_window = (plot_start,plot_end)

        fig = figure(figsize = (8,11))
        ax1 = subplot(6,2,1)
        ax2 = subplot(6,2,2,sharex = ax1)
        
        count = 3
        for findex in range(5):
            for pindex in [4,10]:
                if pindex == 4:
                    ax1 = subplot(6,2,1)
                else:
                    ax2 = subplot(6,2,2)
                self.plot_trialtype('Xpos',findex+1,pindex+1,(plot_window[0],plot_window[1]),transform = expan_transform)
                ax3 = subplot(6,2,count,sharex = ax1)
                self.plot_trialtype('L_m_R',findex+1,pindex+1,(plot_window[0],plot_window[1]))
                if pindex == 4:
                    ax3.set_ylabel('l/v =%s ms'%(L_V_lookup[findex+1]))
                ax3.set_ybound(-10,10)
                ax1.set_xbound(float(plot_start),float(plot_end))
                ax1.set_ybound(0,95)
                ax2.set_xbound(float(plot_start),float(plot_end))
                ax2.set_ybound(0,95)
                count +=1

        ax1.set_title('pole =  90')
        ax2.set_title('pole = -90')
        suptitle('Fly%s'%(flynum))
        return fig
    
    def make_ol_ave(self,ave_start,ave_end):
        datastorage = dict()
        #fly = FlyRecord(flynum,repindex,dataroot)
        ave_window = (ave_start,ave_end)
        for findex in range(5):
            for pindex in [4,10]:
                datastorage['Xpos',findex+1,pindex+1] = self.calculate_average('Xpos',findex+1,pindex+1,ave_window,transform = expan_transform)
                datastorage['L_m_R',findex+1,pindex+1] = self.calculate_average('L_m_R',findex+1,pindex+1,ave_window)
        return datastorage
    
##plotting

#def get_phase_trace(L_h,R_h)
#    from scipy import hilbert
#    phases = np.angle(hilbert(get_low_filter(L_h+R_h,500)))
    
    

"""
def plot_ephys_sweep(fly,findex,pindex,sweepnum):
    times = fly['AMsysCh1',findex,pindex][sweepnum].times[-12000:-6000]
    fig = figure(figsize=(6,12))
    ax1 = subplot(3,1,1)
    plot(times,expan_transform(fly['Xpos',findex,pindex][sweepnum][-12000:-6000]))
    ax2 = subplot(3,1,2,sharex = ax1)
    plot(times,fly['AMsysCh1',findex,pindex][sweepnum][-12000:-6000])
    ax3 = subplot(3,1,3,sharex = ax1)
    plot(times,fly['LeftWing',findex,pindex][sweepnum][-12000:-6000])
    plot(times,fly['RightWing',findex,pindex][sweepnum][-12000:-6000])
    return fig
 """


    
def ts(sweep,start,stop):
    sta_index = argwhere(sweep.times >=start)[0]
    stp_index = argwhere(sweep.times >=stop)[0]
    return neo.AnalogSignal(sweep[sta_index:stp_index],
                            t_start = sweep.times[sta_index],
                            sampling_rate = sweep.sampling_rate)

def get_wbas(L_h,R_h):
    pks = get_wingbeats(L_h+R_h)
    newpks,flips = recondition_peaks(L_h+R_h,pks)
    L_a = np.zeros_like(L_h)
    R_a = np.zeros_like(R_h)
    p0 = newpks[0]
    for pk in newpks[1:]:
        L_a[p0:pk] = L_h[p0]
        R_a[p0:pk] = R_h[p0]
        p0 = pk
    L_a[0:newpks[0]] = L_h[0]
    R_a[0:newpks[0]] = R_h[0]
    L_a[newpks[-1]:] = L_h[newpks[-1]]
    R_a[newpks[-1]:] = R_h[newpks[-1]]
    return L_a,R_a
    
def get_phase_trace(L_h,R_h):
    from scipy.signal import hilbert,find_peaks_cwt
    phases = np.angle(hilbert(get_low_filter(L_h+R_h,500)))
    #pks = find_peaks_cwt(L_h+R_h,np.arange(10,20))
    pks = get_wingbeats(L_h+R_h)
    newpks,flips = recondition_peaks(L_h+R_h,pks)
    phase_shift = np.mean(phases[flips])
    phases = np.mod(np.unwrap(phases)-phase_shift,2*np.pi)
    return phases

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
        
def get_wingbeats(wb_signal):
    thresh = 0.5
    from scipy.signal import medfilt
    detrend = np.array(wb_signal)-medfilt(wb_signal,151)
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
 
def get_spiketrain(sweep):
    thresh = 10.0
    from scipy.signal import medfilt
    wl = 25
    wr = 20
    detrend = np.array(sweep)-medfilt(sweep,35)
    deltas = np.diff(np.array(detrend>thresh,dtype = 'float'))
    starts = np.argwhere(deltas>0.5)
    stops = np.argwhere(deltas<-0.5)
    if starts[0] > stops[0]:
        stops = stops[1:]
    if stops[-1] < starts[-1]:
        starts = starts[:-1]
    intervals = np.hstack((starts,stops))
    peaks = [np.argmax(sweep[sta:stp])+sta for sta,stp in intervals]
    waveforms = [sweep[pk-wl:pk+wr] for pk in peaks]
    sweep.sampling_period.units = 's'
    pk_tms = sweep.times[array(peaks)]
    #pk_tms = pq.Quantity([pk*sweep.sampling_period + sweep.t_start for pk in peaks])
    spike_train = neo.SpikeTrain(pk_tms,
                                sweep.t_stop,
                                sampling_rate = sweep.sampling_rate,
                                waveforms = waveforms,
                                left_sweep = wl*sweep.sampling_period,
                                t_start = sweep.t_start,
                                pk_ind = peaks)
    return spike_train

"""                
def get_spiketrain(sweep):
    get the spiketrain associated with an asig return the spike train in 
    neo's spiketrain format
    #sweep = array(asig)
    #first filter the sweep
    filtered = np.array(sweep) - medfilt(sweep,51)
    #find the peaks - come up with something better than hard coded params
    pks = find_peaks_cwt(filtered,np.arange(30,45),min_snr = 0.05)[3:]
    #allocate memory for an array of offsets corresponding to the index
    #difference of the max amplitude from the index returned (within window)
    offsets = np.zeros_like(pks)
    #datamtrx = np.zeros(shape= (60,len(pks)))
    waveforms = list()
    for i,pk in enumerate(pks):
        offset = np.argmax(filtered[pk-30:pk+30])-30
        #datamtrx[:,i] = filtered[pk-30+offset:pk+30+offset]
        waveforms.append(sweep[pk-30+offset:pk+30+offset])
        offsets[i] = offset
        
    pk_ind = [p+offset for p,offset in zip(pks,offsets)]
    sweep.sampling_period.units = 's'
    pk_tms = pq.Quantity([pk*sweep.sampling_period + sweep.t_start for pk in pk_ind])
    spike_train = neo.SpikeTrain(pk_tms,
                                sweep.t_stop,
                                sampling_rate = sweep.sampling_rate,
                                waveforms = waveforms,
                                left_sweep = 30*sweep.sampling_period,
                                t_start = sweep.t_start,
                                pk_ind = pk_ind)
    return spike_train
"""


def sort_spikes(wv_mtrx,M):
    from scipy.linalg import svd
    from scipy.cluster.vq import kmeans2
    import sklearn
    from sklearn import cluster, datasets
    from sklearn.metrics import euclidean_distances
    from sklearn.neighbors import kneighbors_graph
    from sklearn.preprocessing import StandardScaler
    from sklearn import svm
    from sklearn.covariance import EllipticEnvelope
    import pywt
    from scipy import stats
    
    p2p = np.max(wv_mtrx,axis = 1) -np.min(wv_mtrx,axis = 1)
    p2pt = np.argmax(wv_mtrx,axis = 1) - np.argmin(wv_mtrx,axis = 1)
    
    
    ##medsweep erro
    wv_med = np.median(wv_mtrx,axis = 0)
    err_vec = np.sum(np.sqrt((wv_mtrx-wv_med)**2),axis = 1)
    err_vec /= np.max(err_vec)
    print(shape(err_vec))
    print(shape(p2p))
    
    wave_dff = np.diff(wv_mtrx,axis = 1)
    
    
    dp2p = np.max(wave_dff,axis = 1) -np.min(wave_dff,axis = 1)
    dp2pt = np.argmax(wave_dff,axis = 1) - np.argmin(wave_dff,axis = 1)
    #print(shape(wv_mtrx))
    
    
    wtr = vstack([hstack(pywt.wavedec(wv_mtrx[x,:],'db9',level = 3)) for x in range(shape(wv_mtrx)[0])])
    wtr2 = vstack([hstack(pywt.wavedec(wv_mtrx[x,:],'haar',level = 3)) for x in range(shape(wv_mtrx)[0])])
    #wv_mtrx = vstack([pywt.dwt(wv_mtrx[x,:],'db2')[1] for x in range(shape(wv_mtrx)[0])])
    #print(shape(wv_mtrx))
    wv_mean = np.mean(wv_mtrx)
    datamtrx = wv_mtrx-wv_mean
    U,s,Vt = svd(datamtrx,full_matrices=False)
    V = Vt.T

    ind = np.argsort(s)[::-1]
    U = U[:,ind]
    s = s[ind]
    V = V[:,ind]
    #print shape(wtr)
    
    
    #print(shape(array([p2p]).T))
    #print(shape(U))
    #print shape(wtr)
    #features = np.concatenate((wtr,array([p2p]).T,array([p2pt]).T,array([dp2p]).T,array([dp2pt]).T,U),axis = 1)
    #features = np.concatenate((array([p2p]).T,array([p2pt]).T,array([dp2p]).T,array([dp2pt]).T,U),axis = 1)
    features = np.concatenate((array([p2p]).T,array([err_vec]).T,array([p2pt]).T,wtr[:,:3],U[:,:3]),axis = 1)
    #features = np.concatenate((array([p2p]).T,array([p2pt]).T,wtr[:,:3],U[:,:3]),axis = 1)
    #dbscan = cluster.DBSCAN(eps=1.0)
    X = StandardScaler().fit_transform(features[:,:2])
    
    #outliers_fraction = 0.25
    #clsf = svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05,
    #                                 kernel="rbf", gamma=0.1)
    
    
                                            
           
    #X = features[:,:5]
    #clsf.fit(X[:,:2])
    #y_pred = clsf.decision_function(X).ravel()
    #threshold = stats.scoreatpercentile(y_pred,
    #                                        100 * outliers_fraction)
    #idx = y_pred > threshold
    #dbscan.fit(X[:,:5])
    #idx = dbscan.labels_.astype(np.int)
    
    #features = Ur
    #print(shape(features))
    #print shape(U)
    #M = 2
    #M = np.array([[-0.72394844,0.6905216 ],[ 0.50694304,-0.48353598]])
    es, idx = kmeans2(X[:,:4],M)
    print es
    #es, idx = kmeans2(features[:,:3],2,minit = 'points')
    #es, idx = kmeans2(features[:,:8],4)
    return idx,features,U

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

def phase_raster(event_times_list,phase_list):
    ax = plb.gca()
    for ith,trial in enumerate(zip(event_times_list,phase_list)):
        colors = plb.cm.jet(trial[1]/(2*np.pi))
        plb.vlines(trial[0],ith + 0.5, ith+ 1.5,color = colors)
    return ax

def raster(event_times_list, color = 'k'):
    """
    Creates a raster plot
 
    Parameters
    ----------
    event_times_list : iterable
                       a list of event time iterables
    color : string
            color of vlines
 
    Returns
    -------
    ax : an axis containing the raster plot
    
    from http://scimusing.wordpress.com/2013/05/06/making-raster-plots-in-python-with-matplotlib/
    """
    
    ax = plt.gca()
    for ith, trial in enumerate(event_times_list):
        plt.vlines(trial, ith + .5, ith + 1.5, color=color)
    plt.ylim(.5, len(event_times_list) + .5)
    return ax

#code to plot l/|v| vs tcol - need to fix so time runs backwards.
#filtered = [psf.get_low_filter(amtrx['L_m_R',x,5],50) for x in range(1,6)]
#filtered2 = [psf.get_low_filter(amtrx['L_m_R',x,11],50) for x in range(1,6)]
#amtrx,stdemtrx,groupmtrx = psf.calculate_groupwise_means(flylist = [3,4,5,6,7,9])
#plot([5,20,50,100,200],[x.times[argmin(x)] for x in filtered],'o',ms = 20)
#[plot(amtrx['L_m_R',x,11][::100].times,amtrx['L_m_R',x,11][::100]) for x in range(1,6)]