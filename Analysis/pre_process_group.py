import expansion_lib as explib
import os,sys
import pylab as plb

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
                 

def fly_save(flynum,repindex,rootpath):
    import cPickle as cpkl
    print('loading data')
    fly = explib.FlyRecord(flynum,repindex,rootpath)
    print('calculating open loop averages')
    ave_start = fly.epoch*-1
    ave_end = fly.tail
    datastorage = explib.make_ol_ave(fly,ave_start,ave_end)
    print('saving data to: open_loop_fly%s.cpkl'%(flynum))
    fi = open(os.path.join(rootpath,'open_loop_fly%s.cpkl'%(flynum)),'wb')
    cpkl.dump(datastorage,fi)
    print('closing file')
    fi.close()
    print('calculating closed loop histograms')
    cl_hist = explib.get_closed_loop_histogram(fly,savefig = False)
    print('saving closed histogram data to: closed_loop_fly%s.cpkl'%(flynum))
    fi = open(os.path.join(rootpath,'closed_loop_fly%s.cpkl'%(flynum)),'wb')
    cpkl.dump(cl_hist,fi)
    print('closing file')
    fi.close()
    print('making open loop summary')
    fig = explib.plot_fly_summary(fly)
    print('saving fig as: open_loop%s.pdf'%(flynum))
    plb.savefig(os.path.join(rootpath,'open_loop_fly%s.pdf'%(flynum)))
    plb.close(fig)
    print('making closed loop summary')
    fig = explib.plot_closed_loop_summary(fly)
    print('saving fig as: closed_loop%s.pdf'%(flynum))
    plb.savefig(os.path.join(rootpath,'closed_loop_fly%s.pdf'%(flynum)))
    plb.close(fig)
    del(fly)

if __name__ == '__main__':
    for nums in sq_group_nums:
        fly_save(nums[0],nums[1],sq_group_path)
 