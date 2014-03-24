import numpy as np

class SpkCollection(object):
    def __init__(self,spike_pool,selection_mask,params):
        self.spike_pool = spike_pool
        self.selection_mask = selection_mask
        self.params = params
        
    def collection_wvmtrx(self):
        return self.spike_pool.wv_mtrx[self.collection_ind(),:]
        
    def collection_ind(self):
        return self.spike_pool.spk_ind[np.argwhere(self.selection_mask)[:,0]]
        
class SpkSelector(SpkCollection):  
    def __init__(self,spike_pool,selection_mask,input_mtrx,params):
        super(SpkSelector,self).__init__(spike_pool,selection_mask,params)
        self.input_mtrx = input_mtrx
        self.labels = np.zeros(np.shape(selection_mask),dtype = 'S20')
        
    def mask_from_labels(self,select_labels):
        select_labels = np.array(select_labels)
        mask = np.in1d(self.labels,select_labels)
        return mask
        
    def ind_from_labels(self,select_labels):
        mask = self.mask_from_labels(select_labels)
        return self.spike_pool.spk_ind[np.argwhere(mask)[:,0]]
        
    def wv_mtrx_from_labels(self,select_labels):
        return self.spike_pool.wv_mtrx[self.ind_from_labels(select_labels)]
        
    def select(self):
        pass

class SpkTransformer(SpkCollection):
    def __init__(self,spike_pool,selection_mask,params):
        super(SpkTransformer,self).__init__(spike_pool,selection_mask,params)
        self.trnsmtrx = np.zeros((np.shape(self.spike_pool.wv_mtrx)[0],
                                 self.params['trans_dims']))
        
    def collection_trnsmtrx(self):
        return self.trnsmtrx[self.collection_ind(),:]
    
    def transform(self):
        pass

class SampleRandomSeq(SpkSelector):
    def select(self,seq_len,n_seq):
        seq_len = self.params['seq_len']
        n_seq = self.params['n_seq']
        import random
        idx = self.collection_ind()
        self.seq_starts = random.sample(idx[::seq_len],n_seq)
        self.seq_starts.sort()
        for i,st in enumerate(self.seq_starts):
            self.labels[st:st+seq_len] = 'seq%s'%(i)

class SpectralCluster(SpkSelector):
    def select(self):
        from sklearn.metrics import euclidean_distances
        wv_mtrx = self.collection_wvmtrx()
        #print "computing distances"
        #distances = euclidean_distances(wv_mtrx,wv_mtrx)
        from sklearn.cluster import SpectralClustering
        self.est = SpectralClustering(n_clusters=2,
                                      affinity="nearest_neighbors")
        #print "fitting"
        self.est.fit(wv_mtrx)
        labels = self.est.labels_
        self.labels[self.collection_ind()] = labels
        
class P2PTransform(SpkTransformer):
    def transform(self):
        wv_mtrx = self.collection_wvmtrx()
        p2p = np.max(wv_mtrx,axis = 1) -np.min(wv_mtrx,axis = 1)
        p2pt = np.argmax(wv_mtrx,axis = 1) - np.argmin(wv_mtrx,axis = 1)
        print(np.shape(p2p))
        print(np.shape(p2pt))
        self.trnsmtrx = np.hstack((np.array([p2p]).T,np.array([p2pt]).T))
    
    
class KMeansCluster(SpkSelector):
    def select(self):
        from sklearn.cluster import KMeans
        X = self.input_mtrx[self.collection_ind()]
        self.est = KMeans(n_clusters= self.params['kmeans_nc'],
                          init = self.params['init'])
        self.est.fit(X)
        labels = self.est.labels_
        self.labels[self.collection_ind()] = labels
        
class PCATransform(SpkTransformer):
    def transform(self):
        from sklearn import decomposition
        wv_mtrx = self.collection_wvmtrx()
        self.est = decomposition.PCA(n_components=self.params['trans_dims'],
                                     whiten = self.params['pca_whiten'])
        self.est.fit(wv_mtrx)
        self.trnsmtrx[self.collection_ind(),:] = self.est.transform(wv_mtrx)
        

class DBSCANCluster(SpkSelector):
    def select(self):
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        X = self.input_mtrx[self.collection_ind()]
        #X = StandardScaler().fit_transform(X)
        self.est = DBSCAN(eps = self.params['eps'],
                          min_samples = self.params['min_samples'])
        self.est.fit(X)
        labels = self.est.labels_
        self.labels[self.collection_ind()] = labels.astype(int)

class WTR(SpkTransformer):
    def __init__(self,spike_pool,selection_mask,params):        
        super(SpkTransformer,self).__init__(spike_pool,selection_mask,params)
        #self.trnsmtrx = np.zeros((np.shape(self.spike_pool.wv_mtrx)[0],self.params['trans_dims']))
        
    def transform(self):
        import pywt
        X = self.collection_wvmtrx()
        wavelet = pywt.Wavelet(self.params['wavelet'])
        
        n_samples = X.shape[1]
        n_spikes = X.shape[0]
        
        def full_coeff_len(datalen, filtlen, mode):
            max_level = pywt.dwt_max_level(datalen, filtlen)
            total_len = 0
            for i in xrange(max_level):
                datalen = pywt.dwt_coeff_len(datalen, filtlen, mode)
                total_len += datalen 
            return total_len + datalen
        
        n_features = full_coeff_len(n_samples, wavelet.dec_len, 'sym')
        
        est_mtrx = np.zeros((n_spikes,n_features))
        self.trnsmtrx = np.zeros((np.shape(self.spike_pool.wv_mtrx)[0],n_features))
        for i in xrange(n_spikes):
            tmp = np.hstack(pywt.wavedec(X[i, :], wavelet, 'sym'))
            est_mtrx[i, :] = tmp
        self.trnsmtrx[self.collection_ind(),:] = est_mtrx

class WvltPCA(SpkTransformer):
    def transform(self):
        wvlt_params = {'wavelet':'db1'}
        wt = WTR(self.spike_pool,self.selection_mask,wvlt_params)
        wt.transform()
        from sklearn import decomposition
        wv_mtrx = wt.trnsmtrx[self.collection_ind(),:]
        self.est = decomposition.PCA(n_components=self.params['trans_dims'],
                                     whiten = self.params['pca_whiten'])
        self.est.fit(wv_mtrx)
        self.trnsmtrx[self.collection_ind(),:] = self.est.transform(wv_mtrx)
        
class MedTrans(SpkTransformer):
    def transform(self,inputmtrx = None):
        if not(inputmtrx == None):
            wv_mtrx = inputmtrx[self.collection_ind(),:]
        else:
            wv_mtrx = self.collection_wvmtrx()
        wv_mtrx = self.collection_wvmtrx()
        self.wv_med = np.median(wv_mtrx,axis = 0)
        self.resmtrx = wv_mtrx-self.wv_med
        err_vec = np.sum(np.sqrt(np.square(self.resmtrx)),axis = 1)
        #err_vec /= np.max(err_vec)
        self.trnsmtrx[self.collection_ind(),:] = err_vec[:,np.newaxis]

class ThreshSelector(SpkSelector):
    def select(self,thresh = 0.5):
        X = self.input_mtrx[self.collection_ind()]
        self.labels[self.collection_ind()] = np.array(X > thresh,dtype = int)
        
class MBKMeansCluster(SpkSelector):
    def select(self):
        from sklearn.cluster import MiniBatchKMeans
        X = self.input_mtrx[self.collection_ind()]
        self.est = MiniBatchKMeans(n_clusters= self.params['kmeans_nc'],
                          init = self.params['init'],batch_size = 20)
        self.est.fit(X)
        labels = self.est.labels_
        self.labels[self.collection_ind()] = labels
        
class GMMCluster(SpkSelector):
    def select(self):
        from sklearn import mixture
        X = self.input_mtrx[self.collection_ind()]
        self.est = mixture.DPGMM(n_components=3)
        self.est.fit(X)
        labels = self.est.predict(X)
        self.labels[self.collection_ind()] = labels

class PCATransform2(SpkTransformer):
    def transform(self,inputmtrx = None):
        from sklearn import decomposition
        if not(inputmtrx == None):
            wv_mtrx = inputmtrx[self.collection_ind(),:]
        else:
            wv_mtrx = self.collection_wvmtrx()
        self.est = decomposition.PCA(n_components=self.params['trans_dims'],
                                     whiten = self.params['pca_whiten'])
        self.est.fit(wv_mtrx)
        self.trnsmtrx[self.collection_ind(),:] = self.est.transform(wv_mtrx)
        