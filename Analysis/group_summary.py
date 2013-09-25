import expansion_lib as explib
import os,sys
import pylab as plb
import numpy as np
import trace_tools as trt

rootsym = {'darwin':'/'}
rootlist = [rootsym[sys.platform],'Users','psilentp','Dropbox','Data','behavioral_pilot']

ln_group_path = os.path.join(*rootlist + ['expansion_critical_angle_ln'])
sq_group_path = os.path.join(*rootlist + ['expansion_critical_angle_sq'])


ln_group_nums = [(2,0),(3,0),(4,0),(5,0),(6,0),
                 (7,1),(8,0),(9,0),(10,0),(11,0),
                 (12,0)]

sq_group_nums = [(2,0),(3,1),(4,0),(5,0),(6,0),
                 (7,0),(8,0),(9,0),(10,0),(11,0),
                 (12,0),(13,0),(14,1),(15,1),(16,0),
                 (17,0)]
                 

def calculate_groupwise_means(flylist,rootpath):
    import cPickle as cpkl
    group_matrix = dict()
    average_matrix = dict()
    stde_matrix = dict()
    #flylist = [1,2,3,4,5,6,7,9,10,11,12]
    for findex in range(5):
            for pindex in[4,10]:
                group_matrix['Xpos',findex+1,pindex+1] = list()
                group_matrix['L_m_R',findex+1,pindex+1] = list()
                average_matrix['Xpos',findex+1,pindex+1] = list()
                average_matrix['L_m_R',findex+1,pindex+1] = list()  
                stde_matrix['Xpos',findex+1,pindex+1] = list()
                stde_matrix['L_m_R',findex+1,pindex+1] = list()
    #load the group matrix with fly data
    for flynum in flylist:
        fi = open(os.path.join(rootpath,'open_loop_fly%s.cpkl'%(flynum)),'rb')
        fly_data = cpkl.load(fi)
        print("loading fly%s"%(flynum))
        for findex in range(5):
            for pindex in[4,10]:
                group_matrix['Xpos',findex+1,pindex+1].append(fly_data['Xpos',findex+1,pindex+1])
                group_matrix['L_m_R',findex+1,pindex+1].append(fly_data['L_m_R',findex+1,pindex+1])
    for findex in range(5):
        for pindex in [4,10]:
            print('averaging cell %s %s'%(findex,pindex))
            average_matrix['Xpos',findex+1,pindex+1] = trt.get_signal_mean(group_matrix['Xpos',findex+1,pindex+1])
            average_matrix['L_m_R',findex+1,pindex+1] = trt.get_signal_mean(group_matrix['L_m_R',findex+1,pindex+1])
            stde_matrix['Xpos',findex+1,pindex+1] = trt.get_signal_stderr(group_matrix['Xpos',findex+1,pindex+1],signal_mean = average_matrix['Xpos',findex+1,pindex+1])
            stde_matrix['L_m_R',findex+1,pindex+1] = trt.get_signal_stderr(group_matrix['L_m_R',findex+1,pindex+1],signal_mean = average_matrix['L_m_R',findex+1,pindex+1])
    return average_matrix,stde_matrix,group_matrix

def get_cl_data(flylist,rootpath):
    import cPickle as cpkl
    return [cpkl.load(open(os.path.join(rootpath,'closed_loop_fly%s.cpkl'%(flynum)),'rb')) for flynum in flylist]

def plot_group_summary(amtrx,cldata):
    from pylab import cm
    for c_index,seq_index in zip(np.linspace(0.5,1,5),range(1,6)):
        ax = plb.subplot(2,2,1)
        plb.plot(amtrx['Xpos',seq_index,11][::100].times,amtrx['Xpos',seq_index,11][::100],color = cm.Blues(c_index))
        plb.subplot(2,2,3,sharex = ax)
        plb.plot(amtrx['L_m_R',seq_index,11][::100].times,amtrx['L_m_R',seq_index,11][::100],color = cm.Blues(c_index))
        plb.plot(amtrx['L_m_R',seq_index,5][::100].times,amtrx['L_m_R',seq_index,5][::100],color = cm.Blues(c_index))
    
    filtered = [trt.get_low_filter(amtrx['L_m_R',x,5],20) for x in range(1,6)]
    filtered2 = [trt.get_low_filter(amtrx['L_m_R',x,11],20) for x in range(1,6)]
    plb.subplot(2,2,2)
    plb.plot([5,20,50,100,200],[x.times[np.argmin(x)] for x in filtered],'o',ms = 5)
    plb.plot([5,20,50,100,200],[x.times[np.argmax(x)] for x in filtered2],'o',ms = 5)
    plb.subplot(2,2,4,polar = True)
    [plb.plot(np.linspace(0,2*np.pi,94),x[0],color = 'k') for x in cldata]
    ax.set_xbound(-0.5,0.2)

if __name__ == '__main__':
    global amtrx
    flylist = [x[0] for x in sq_group_nums]
    amtrx,stdemtrx,groupmtrx = calculate_groupwise_means(flylist = flylist,rootpath = sq_group_path)
    
    
    
    
