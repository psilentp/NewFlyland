from pylab import *

def plot_clusters(sp,selector):
    figure(figsize=(2,4))
    mask = nonzero(selector.labels)
    ax1 = subplot(3,1,1)
    dt = 1/20000.
    samples = shape(sp.wv_mtrx)[1]
    times = linspace(0,samples*dt,samples)
    #print times
    for wf,lb in zip(sp.wv_mtrx[:100000:50,:],selector.labels[:100000:50]):
        try:
            ax2 = subplot(3,1,int(lb)+1,sharex = ax1,sharey = ax1)
            color_lookup = {'':'b','-1':'k','0':'r','1':'g','2':'y','3':'c','4':'k'}
            color = color_lookup[lb]
            #print col
            plot(times,wf,color = color,alpha = 0.2)
            ax2.set_ylabel('uV')
            ax2.set_xlabel('time (s)')
        except ValueError:
            pass
        
            #print lb
    ax1.set_xbound(0,times[-1])
    ax1.set_ybound(np.min(sp.wv_mtrx),np.max(sp.wv_mtrx))
    ax1.set_ylabel('uV')
    
            
def plot_select_trains(sp,masks):
    n_masks = len(masks)
    ax0 = subplot(n_masks+2,1,1)
    for wv in sp.waveforms[:900]:
        plot(wv.times,wv,color = 'k')
    for i,mask in enumerate(masks):
        subplot(n_masks+1,1,i+2,sharex = ax0,sharey = ax0)
        ind = argwhere(mask[:900])
        [plot(sp.waveforms[x].times,sp.waveforms[x],lw = 3, alpha = 0.3,color = 'b') for x in ind]