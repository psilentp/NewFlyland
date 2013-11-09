import numpy as np

class SpkSorter(object):
    def __init__(self,spike_pool,selection_mask,params):
        self.spike_pool = spike_pool
        self.selection_mask = selection_mask
        self.params = params
        self.labels = np.zeros(np.shape(selection_mask),dtype = 'S20')
    
    def selected_wvmtrx(self):
        return self.spike_pool.wv_mtrx[np.argwhere(self.selection_mask)[:,0],:]
        
    def selected_ind(self):
        return self.spike_pool.spk_ind[np.argwhere(self.selection_mask)[:,0]]
        
    def mask_from_labels(self,select_labels):
        select_labels = np.array(select_labels)
        mask = np.in1d(self.labels,select_labels)
        return mask
        
    def sort(self):
        pass


class SampleRandomSeq(SpkSorter):
    def sort(self,seq_len,n_seq):
        seq_len = self.params['seq_len']
        n_seq = self.params['n_seq']
        import random
        idx = self.selected_ind()
        seq_starts = random.sample(idx[::seq_len],n_seq)
        seq_starts.sort()
        for i,st in enumerate(seq_starts):
            self.labels[st:st+seq_len] = 'seq%s'%(i)
           
class PCACluster(SpkSorter):
    def sort(self):
        from sklearn.cluster import KMeans
        from sklearn import decomposition
        wv_mtrx = self.selected_wvmtrx()
        pca = decomposition.PCA(n_components=self.params['pca_components'],
                                whiten = self.params['pca_whiten'])
        pca.fit(wv_mtrx)
        X = pca.transform(wv_mtrx)
        est = KMeans(n_clusters= self.params['kmeans_nc'])
        est.fit(X)
        labels = est.labels_
        self.labels[self.selected_ind()] = labels