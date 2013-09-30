import numpy as np
import pylab as plb

def plot_lowspikes(spiketrains):
    for st in spiketrains:
        lowind = np.argwhere(st.annotations['phases']>4)
        [plb.plot(st.waveforms[x], color = 'k',alpha = 0.5) for x in lowind[:20]]

def get_wingbeats(wb_signal):
    thresh = 0.5
    from scipy.signal import medfilt
    detrend = np.array(wb_signal)-medfilt(wb_signal,151)
    deltas = np.diff(np.array(detrend>thresh,dtype = 'float'))
    starts = np.argwhere(deltas>0.5)
    stops = np.argwhere(deltas<-0.5)
    if starts[0] > stops[0]:
        stops = stops[1:]
    intervals = np.hstack((starts,stops))
    peaks = [np.argmax(wb_signal[sta:stp])+sta for sta,stp in intervals]
    return peaks
    
def get_spikepeaks(signal):
    thresh = 10.0
    from scipy.signal import medfilt
    detrend = np.array(signal)-medfilt(signal,35)
    deltas = np.diff(np.array(detrend>thresh,dtype = 'float'))
    starts = np.argwhere(deltas>0.5)
    stops = np.argwhere(deltas<-0.5)
    if starts[0] > stops[0]:
        stops = stops[1:]
    intervals = np.hstack((starts,stops))
    peaks = [np.argmax(signal[sta:stp])+sta for sta,stp in intervals]
    return peaks