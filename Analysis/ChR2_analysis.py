dataroot = '/Users/psilentp/Dropbox/Data/ChR2/'

from pylab import *
import expansion_lib as expl
fly = expl.FlyRecord(8,1,dataroot)
signal_sweep = fly.signals['AMsysCh1'];
condition_sweep = fly.signals['IN 11'];
edge_sweep = fly.signals['Sync'];
edgepoints = argwhere(diff(array(edge_sweep<1,dtype = int))>0.5)
switch = True
for ep in edgepoints[3500:5000]:
    if condition_sweep[ep+20]>1:
        subplot(2,1,1)
        plot(signal_sweep[ep:ep+200],alpha = 0.1,color = 'k')
    else:
        subplot(2,1,2)
        if switch:
            plot(signal_sweep[ep:ep+200],alpha = 0.1,color = 'k')
        switch = not(switch)
show()
