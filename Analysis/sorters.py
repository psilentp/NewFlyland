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
        seq_starts = random.sample(idx[::seq_len],n_seq)
        seq_starts.sort()
        for i,st in enumerate(seq_starts):
            self.labels[st:st+seq_len] = 'seq%s'%(i)

class KMeansCluster(SpkSelector):
    def select(self):
        from sklearn.cluster import KMeans
        X = self.input_mtrx[self.collection_ind()]
        est = KMeans(n_clusters= self.params['kmeans_nc'])
        est.fit(X)
        labels = est.labels_
        self.labels[self.collection_ind()] = labels
        
class PCATransform(SpkTransformer):
    def transform(self):
        from sklearn import decomposition
        wv_mtrx = self.collection_wvmtrx()
        pca = decomposition.PCA(n_components=self.params['trans_dims'],
                                whiten = self.params['pca_whiten'])
        pca.fit(wv_mtrx)
        self.trnsmtrx[self.collection_ind(),:] = pca.transform(wv_mtrx)
        