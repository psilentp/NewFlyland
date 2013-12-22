import numpy as np

pca_trans = {'trans_dims':3,'pca_whiten':True}

km_cluster = {'kmeans_nc':2,'init':'k-means++'}
randseq = {'seq_len':100,'n_seq':10}


pool_params = {'spk_pool_thresh':10.0,
               'spk_window_left' : 25,
               'spk_window_right' : 20,
               'spk_pool_filter_window' : 35}


pattern_xnum = 768
deg_per_xstp = 360./pattern_xnum
ol_volts_per_deg = 10.0/(pattern_xnum*deg_per_xstp)
cl_volts_per_rad = 5/(np.pi)
expan_transform = lambda x:x/ol_volts_per_deg
fix_transform = lambda x:x/cl_volts_per_rad