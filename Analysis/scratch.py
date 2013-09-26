import expansion_lib as expl
from scipy.signal import medfilt,find_peaks_cwt
import numpy as np
import pylab as plb
from scipy.cluster.vq import *

def grid_objects(alist, blist):
    for i in range(len(alist)):
        for j in range(len(alist[i])):
            yield(alist[i][j], blist[i][j])
            
#dataroot = '/Users/psilentp/Dropbox/Data/FirstEphys/'
dataroot = '/Users/psilentp/Dropbox/Data/LeftRight/'

fly = expl.FlyRecord(2,0,dataroot)
sweep = fly['AMsysCh1',2,5][4][-150000:-6000]

filtered = np.array(sweep) - medfilt(sweep,51)
pks = find_peaks_cwt(filtered,np.arange(9,15))
offsets = np.zeros_like(pks)
datamtrx = np.zeros(shape= (60,len(pks[3:])))

for i,pk in enumerate(pks[3:]):
    offset = np.argmax(filtered[pk-30:pk+30])-30
    datamtrx[:,i] = filtered[pk-30+offset:pk+30+offset]
    offsets[i+3] = offset
        #plb.plot(filtered[pk-40+offset:pk+20+offset],color = 'k', alpha = 0.1)

from scipy.linalg import svd
U,s,Vt = svd(datamtrx,full_matrices=False)
V = Vt.T

ind = np.argsort(s)[::-1]
U = U[:,ind]
s = s[ind]
V = V[:,ind]

features = V
es, idx = kmeans2(features[:,0:4],2,iter=50)
colors = ([([1,0,1],[1,0,0],[0,0,1],[0,1,1])[i] for i in idx])


#plb.plot(V[:,0],V[:,1],'o')
#plb.show()

#V = np.dot(U,Vt) 
#es, idx = kmeans2(np.array(zip(V[:,0],V[:,1],V[:,2],V[:,3],V[:,4],V[:,5])),4)
#    es, idx = kmeans2(features[:,0:4],4,iter=50)
#    colors = ([([1,0,1],[1,0,0],[0,0,1],[0,1,1])[i] for i in idx])
#plb.scatter(V[:,0],V[:,1], c=colors)



plb.figure()
plb.plot(filtered,color = 'k')
for i,pk in enumerate(pks[3:]):
    plb.plot(np.linspace(pk-30+offsets[i+3],pk+30+offsets[i+3],60),datamtrx[:,i],color = colors[i])


plb.figure()
count = 0
for feature1 in range(4):
    for feature2 in range(4):
        plb.subplot(4,4,count + 1)
        plb.scatter(features[:,feature1],features[:,feature2], c=colors)
        count += 1




plb.figure()
for group,c,i in zip(idx,colors,range(np.shape(datamtrx)[1])):
    print group,c,i
    if group==0:
        plb.subplot(4,1,1)
    if group==1:
        plb.subplot(4,1,2)
    #if group==2:
    #    plb.subplot(4,1,3)
    #if group==3:
    #    plb.subplot(4,1,4)
    plb.plot(datamtrx[:,i],color = c,alpha = 0.1)

plb.show()
