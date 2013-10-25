dataroot = '/Users/psilentp/Dropbox/Data/ChR2/'

from pylab import *
import expansion_lib as expl
fly = expl.FlyRecord(6,2,dataroot)
signal_sweep = fly.signals['AMsysCh1'];condition_sweep = fly.signals['IN 11'];edge_sweep = fly.signals['Sync'];edgepoints = argwhere(diff(array(edge_sweep<1,dtype = int))>0.5)
for ep in edgepoints[:2000]:
    if condition_sweep[ep+20]>1:
        subplot(2,1,1)
        plot(signal_sweep[ep:ep+400],alpha = 0.1,color = 'k')
    else:
        subplot(2,1,2)
        plot(signal_sweep[ep:ep+400],alpha = 0.1,color = 'b')
show()
