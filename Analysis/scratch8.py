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

fly.extract_spike_pool()
st = fly.processed_signals['spike_pool']

rand_sorter = srtr.SampleRandomSeq(st,ones_like(st),defaults.randseq)
rand_sorter.sort(100,10)
rand_mask = rand_sorter.mask_from_labels(['seq'+str(x) for x in range(10)])
pca_sorter = srtr.PCACluster(st,rand_mask,defaults.pca_cluster)
pca_sorter.sort()

figure()
for wf,lb in zip(pca_sorter.selected_wvmtrx(),pca_sorter.labels[pca_sorter.selected_ind()]):
    subplot(2,1,lb)
    plot(wf,color = cm.jet(lb.astype(np.float)/2.0),alpha = 0.2)