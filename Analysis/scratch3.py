import expansion_lib as expl
import numpy as np

def get_phase_trace(L_h,R_h):
    from scipy.signal import hilbert,find_peaks_cwt
    phases = np.angle(hilbert(expl.get_low_filter(L_h+R_h,500)))
    pks = find_peaks_cwt(L_h+R_h,np.arange(10,20))
    newpks = recondition_peaks(L_h+R_h,pks)
    phase_shift = np.mean(phases[newpks])
    phases = np.mod(np.unwrap(phases)-phase_shift,2*np.pi)
    return phases

def recondition_peaks(signal,pks):
    newpks = list()
    for pk in pks:
        offset = np.argmax(signal[pk-10:pk+10])
        newpks.append(pk-10+offset)
    return newpks