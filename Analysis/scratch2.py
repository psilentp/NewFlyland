import expansion_lib as expl
import pylab as plb

dataroot = '/Users/psilentp/Dropbox/Data/LeftRight/'

fly = expl.FlyRecord(2,0,dataroot)

print('plot spike rasters for expansion from left vs right -1 to 0.2 sec')
sweeps = [expl.ts(s,-1,0.2) for s in fly['AMsysCh1',2,5]]
print('calculating spiketrains for first set of sweeps')
spiketrains = [expl.get_spiketrain(s) for s in sweeps]

plb.figure()
ax0 = plb.subplot(2,1,2, sharex)
#plot behavior
behv = expl.get_signal_mean([expl.ts(s,-1,0.2) for s in fly['L_m_R',2,5]])
plb.plot(behv.times,behv)

#plot raster
ax1 = plb.subplot(2,1,2, sharex = ax0)
expl.raster(spiketrains)

sweeps2 = [expl.ts(s,-1,0.2) for s in fly['AMsysCh1',2,11]]
print('calculating spiketrains for second set of sweeps')
spiketrains2 = [expl.get_spiketrain(s) for s in sweeps2]
expl.raster(spiketrains2)
behv = expl.get_signal_mean([expl.ts(s,-1,0.2) for s in fly['L_m_R',2,11]])
plb.plot(behv.times,behv)

plb.figure()
ax0 = plb.subplot(2,1,1)
#plot behavior
behv2 = expl.get_signal_mean([expl.ts(s,-1,0.2) for s in fly['L_m_R',2,5]])
plb.plot(behv2.times,behv2)

#plot raster
ax1 = plb.subplot(2,1,2, sharex = ax0)
expl.raster(spiketrains2)