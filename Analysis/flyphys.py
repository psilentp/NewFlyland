#script to parse the data for a single fly - load % cut up the .abf and matlab format and save the mean traces in a python friendly format.
import os
import neo
import scipy.io as sio
import quantities as pq
from pylab import *
import numpy as np
from scipy.signal import medfilt,find_peaks_cwt
import pylab as plb

from signal_tools import *

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
    """data model for a single ephys recording, trial times
    and types are encoded in the .mat file associated with the .abf file"""
    def __init__(self,
                fly_number,
                axon_file_path,
                mat_file_path,
                clepoch =  pq.Quantity(5,'s'),
                olepoch = pq.Quantity(1.2,'s'),
                min_wbf = pq.Quantity(150.0,'Hz'),
                exp_note_path=None,
                post_process_path = None):
                
        self.fly_number = fly_number
        self.axon_file_path = axon_file_path
        self.mat_file_path = mat_file_path
        self.exp_note_path = exp_note_path
        self.post_process_path = post_process_path
        self.clepoch = clepoch
        self.olepoch = olepoch
        self.tail = pq.Quantity(.4,'s')
        self.processed_signals = dict()
        self.wb_processing_params = dict()
        self.spk_sorting_params = dict()
        self.sorted_spikes = dict()
        self.load_data()
        
        self.left_window = int(self.clepoch/self.dt)
        self.right_window = int(self.olepoch/self.dt + self.tail/self.dt)
        self.min_wbf = min_wbf
        
    def load_data(self):
        importer = neo.io.AxonIO(filename = self.axon_file_path)
        block = importer.read_block()
        hdr = importer.read_header()
        #holds the asignals
        seg = block.segments[0]
        #get and sort the trial presentation record
        trial_matrix = sio.loadmat(self.mat_file_path)['datarecord'][0]
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
    
    def __getitem__(self,k):
        """allow slicing by according to: 
        self[function_index,position_index,trial_num,'SignalString'] 
        fix: make sure that we sqeeze out a singelton list"""
        function_index = k[0]
        position_index = k[1]
        sli = k[2]
        sigkey = k[3]
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
        return retlist
    
    def get_wbf(self,start_ind,end_ind):
        duration = (end_ind-start_ind)*self.signals['Sync'].times[1]
        wingbeats = len(where(diff(where(self.signals['Sync'][start_ind:end_ind]<1,1,0))>0.5)[0])
        freq =  wingbeats/duration
        return freq
     
    def calculate_average(self, 
                          signal_key, 
                          function_index, 
                          position_index,
                          ave_range,
                          transform = None):
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

class FlyController(object):
    
    def __init__(self,fly):
        self.fly = fly
        self.verbose = True
         
    def process_wb_signals(self):
        from scipy.signal import hilbert
        pts_per_wb = 0.005/self.fly.dt #for 200 Hz wb
        L_h = self.fly.signals['LeftWing']
        R_h = self.fly.signals['RightWing']
        arr = L_h+R_h
        if self.verbose:print('calculating flight mask')
        flight_mask = window_stdev(L_h+R_h,int(5*pts_per_wb)) < 0.5 #window apx 5 wb
        if self.verbose:print('calculating stroke phase')
        phases = np.angle(hilbert(get_low_filter(L_h+R_h,500)))
        if self.verbose:print('calculating wb reference features')
        pks = get_wingbeats(L_h+R_h)
        newpks,flips = recondition_peaks(L_h+R_h,pks)
        phase_shift = np.mean(phases[flips])
        if self.verbose:print('re-wraping phase')
        phases = np.mod(np.unwrap(phases)-phase_shift,2*np.pi)
        L_a = np.zeros_like(L_h)
        R_a = np.zeros_like(R_h)
        p0 = newpks[0]
        if self.verbose:print('calculating stroke amplitudes')
        for pk in newpks[1:]:
            L_a[p0:pk] = L_h[p0]
            R_a[p0:pk] = R_h[p0]
            p0 = pk
        L_a[0:newpks[0]] = L_h[0]
        R_a[0:newpks[0]] = R_h[0]
        L_a[newpks[-1]:] = L_h[newpks[-1]]
        R_a[newpks[-1]:] = R_h[newpks[-1]]
        if self.verbose:print('updating processed_signals dict')
        self.fly.processed_signals['wb_phase'] = phases
        self.fly.processed_signals['wb_peaks'] = newpks
        self.fly.processed_signals['wb_flips'] = flips
        self.fly.processed_signals['flight_mask'] = flight_mask
        self.fly.processed_signals['LeftAmp'] = L_a
        self.fly.processed_signals['RightAmp'] = R_a
    
    def extract_spike_pool(self):
        """extract the pool of potential spikes"""
        phys_sig = self.fly.signals['AMsysCh1']
        thresh = self.fly.spk_sorting_params['spk_pool_thresh']
        wl= self.fly.spk_sorting_params['spk_window_left']
        wr= self.fly.spk_sorting_params['spk_window_right']
        filter_window = self.fly.spk_sorting_params['spk_pool_filter_window']
        spike_pool = get_spiketrain(phys_sig,
                                    thresh = thresh,
                                    wl=wl,
                                    wr=wr,
                                    filter_window = filter_window)
        self.fly.processed_signals['spike_pool'] = spike_pool
       

class SpikeSorter(object):
        def __init__(self,**argv):
            self.sorting_params = argv
        
        def sort(self,spike_pool,M=2):
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
            
            spike_sample_indx = np.random.random_integers(0,
                                            len(spike_pool),
                                            self.sorting_params['sample_size'])
                            
            
            wv_mtrx = np.vstack([np.array(spike_pool.waveforms[idx])
                                for idx in spike_sample_indx])
            
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
    
    
            wtr = vstack([hstack(pywt.wavedec(wv_mtrx[x,:],'db9',level = 3))
                                            for x in range(shape(wv_mtrx)[0])])
            wtr2 = vstack([hstack(pywt.wavedec(wv_mtrx[x,:],'haar',level = 3)) 
                                            for x in range(shape(wv_mtrx)[0])])
            wv_mean = np.mean(wv_mtrx)
            datamtrx = wv_mtrx-wv_mean
            U,s,Vt = svd(datamtrx,full_matrices=False)
            V = Vt.T

            ind = np.argsort(s)[::-1]
            U = U[:,ind]
            s = s[ind]
            V = V[:,ind]

            features = np.concatenate((array([p2p]).T,array([err_vec]).T,array([p2pt]).T,wtr[:,:3],U[:,:3]),axis = 1)
            X = StandardScaler().fit_transform(features[:,:2])

            es, idx = kmeans2(X[:,:4],M)
            self.idx = idx
            self.wv_mtrx = wv_mtrx
        
def get_fly_in_rootdir(data_root,fly_number,rep_ind):
        data_dir = data_root + '/Fly%s'%(fly_number)
        data_files = os.listdir(data_dir)
        abf_files = [f for f in data_files if '.abf' in f]
        mat_files = [f for f in data_files if '.mat' in f]
        exp_note_files = [f for f in data_files if '.txt' in f]
        post_process_files = [f for f in data_files if '.cpkl' in f]
        
        axon_file_path = data_dir + '/' + abf_files[rep_ind]
        mat_file_path = data_dir + '/' + mat_files[rep_ind]
        try:
            exp_note_path = data_dir + '/' + exp_note_files[rep_ind]
        except IndexError:
            exp_note_path = None
        try:
            post_process_path = data_dir + '/' + post_process_files[rep_ind]
        except IndexError:
            post_process_path = None
        axon_file_path,mat_file_path,exp_note_path,post_process_path
        fly = FlyRecord(fly_number,
                        axon_file_path = axon_file_path,
                        mat_file_path = mat_file_path,
                        exp_note_path = exp_note_path,
                        post_process_path = post_process_path)
        return fly
        
        
        

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
