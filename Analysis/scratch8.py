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

import time
start = time.clock()
fly = fph.get_fly_in_rootdir(dataroot,2,0)

fly_cntrlr = fph.FlyController(fly)
fly.pool_params = defaults.pool_params

#fly.extract_spike_pool()
#fly.process_wb_signals()
print start - time.clock()
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
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(pca_transformer.collection_trnsmtrx()[:,0],pca_transformer.collection_trnsmtrx()[:,1],pca_transformer.collection_trnsmtrx()[:,2],c = colors)
show()

def plot_rand_selection(fly,rand_selector,lbl_selector):
	from signal_tools import ts
	ax = subplot(10,1,1)
	for si,start in enumerate(rand_selector.seq_starts):
		subplot(10,1,si+1, sharex = ax,sharey = ax)
		sind = rand_selector.ind_from_labels(['seq'+str(si)])
		lbs = lbl_selector.labels[sind]
		color_lookup = {'':'b','0':'r','1':'g'}
		colors = [color_lookup[l] for l in lbs]
		seq_spikes = [rand_selector.spike_pool[i] for i in sind]
		seq_wvfrms = [rand_selector.spike_pool.waveforms[i] for i in sind]
		sweep = ts(fly.signals['AMsysCh1'],seq_spikes[0],seq_spikes[-1])
		st_time = sweep.times[0]
		plot(sweep.times-st_time,sweep,color = 'k')
		[plot(wf.times-st_time,wf,color = c,lw=6,alpha = 0.2) for wf,c in zip(seq_wvfrms,colors)]
		
new_ster = srtr.SpectralCluster(st,rand_mask,pca_transformer.trnsmtrx,defaults.km_cluster)
new_ster.select()
figure()
plot_rand_selection(fly,rand_selector,new_ster)


for wf,lb in zip(spe_0.collection_wvmtrx()[:1000,:],spe_0.labels[spe_0.collection_ind()][:1000]):
    subplot(2,1,int(lb)+1)
    color_lookup = {'':'b','0':'r','1':'g'}
    color = color_lookup[lb]
    plot(wf,color = color,alpha = 0.2)
    
    
#fly.save_data('flydata.cpkl')

"""
rand_wvmtrx = rand_selector.wv_mtrx_from_labels(['seq'+str(x) for x in range(10)])
rand_ind = rand_selector.ind_from_labels(['seq'+str(x) for x in range(10)])
figure()
for wf,lb in zip(rand_wvmtrx,new_ster.labels[rand_ind]):
    subplot(2,1,lb)
    plot(wf,color = cm.jet(lb.astype(np.float)/2.0),alpha = 0.2)
"""