from pylab import *

from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import flyphys as fph
import defaults
import random
from sklearn.cluster import KMeans
import sorters as srtr

dataroot = '/Users/psilentp/Dropbox/Data/LeftRight/'

fly = fph.get_fly_in_rootdir(dataroot,2,0)

fly_cntrlr = fph.FlyController(fly)
fly.pool_params = defaults.pool_params

#fly.extract_spike_pool()
#fly.process_wb_signals()
st = fly.processed_signals['spike_pool']

######random selection##########
rand_selector = srtr.SampleRandomSeq(st,
									 ones_like(st),
									 st.wv_mtrx,
									 defaults.randseq)
rand_selector.select(100,10)
rand_mask = rand_selector.mask_from_labels(['seq'+str(x) for x in range(10)])

#####PCA transformation ##########
pca_transformer = srtr.PCATransform(st,
									rand_mask,
									defaults.pca_trans)
pca_transformer.transform()


#####K-means sorting ##########
km_selector = srtr.KMeansCluster(st,
							   rand_mask,
							   pca_transformer.trnsmtrx,
							   defaults.km_cluster)
km_selector.select()


colors = cm.jet(km_selector.labels[km_selector.collection_ind()].astype(float)/2.0+0.1)
figure()
for wf,lb in zip(km_selector.collection_wvmtrx(),km_selector.labels[km_selector.collection_ind()]):
    subplot(2,1,lb)
    plot(wf,color = cm.jet(lb.astype(np.float)/2.0),alpha = 0.2)

fig = figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_transformer.collection_trnsmtrx()[:,0],pca_transformer.collection_trnsmtrx()[:,1],pca_transformer.collection_trnsmtrx()[:,2],c = colors)
show()
fly.save_data('flydata.cpkl')