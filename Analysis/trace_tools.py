#script to parse the data for a single fly - load % cut up the .abf and matlab format and save the mean traces in a python friendly format.
import os
import neo
import scipy.io as sio
import quantities as pq
from pylab import *
import numpy as np


def get_signal_mean(signal_list):
    """calculate the average signal from a list of signals"""
    #print signal_list
    mean_signal = reduce(lambda x,y:x+y,signal_list)/len(signal_list)
    return mean_signal

def get_signal_stderr(signal_list,signal_mean = None):
    """return the standard error of a group of signals"""
    if signal_mean is None:
        signal_mean = get_signal_mean(signal_list)
    mean_sq_err = get_signal_mean(map(lambda x: (x-signal_mean)**2,signal_list))
    #print sqrt(mean_sq_err)/sqrt(len(signal_list))
    rsig = sqrt(mean_sq_err)/sqrt(len(signal_list))
    rsig.sampling_period = signal_list[0].sampling_period
    rsig.t_start = signal_list[0].t_start
    return rsig

def get_low_filter(signal,Hz):
    """return a low-pass-filterd version of the analog signal"""
    from scipy.signal import filtfilt
    filter_order = 3
    sampfreq = signal.sampling_rate
    filtfreq = pq.Quantity(Hz,'Hz')
    passband_param = float((filtfreq/sampfreq).simplified)
    from scipy.signal import butter
    [b,a]=butter(filter_order,passband_param)
    #ob = cp.copy(signal)
    filt_array = filtfilt(b,a,signal)
    #ob.y = quan.Quantity(filtfilt(b,a,self.y),self.yunits)
    return neo.AnalogSignal(filt_array,
                            units = signal.units,
                            sampling_rate = signal.sampling_rate,
                            t_start = signal.t_start)