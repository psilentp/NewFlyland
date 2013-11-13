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
        