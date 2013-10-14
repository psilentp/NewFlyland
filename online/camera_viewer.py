import wx
import cv2
from cv2 import cv
import pylab as plb
import time
import numpy as np
import copy
from collections import deque

class PlateGui(wx.Frame):

    def __init__(self, *args , **kwds):
        self.frame = wx.Frame.__init__(self,*args, **kwds)
        filemenu= wx.Menu()
        self.framebuffer = deque(maxlen = 10)
        #self.toolbar = self.CreateToolBar(style = wx.TB_BOTTOM | wx.TB_FLAT)
        #qtool = self.toolbar.AddLabelTool(wx.ID_ANY, 'Quit',None)
        #self.toolbar.AddButton(self, wx.ID_OK, "Ok")
        #self.toolbar.Realize()
        #okButton = wx.Button(self, wx.ID_OK, "Save",pos=(0, 500))
        okButton = wx.Button(self, wx.ID_OK, "Save",pos=(0, 800))
        self.Bind(wx.EVT_BUTTON, self.OnClick, okButton)
        
        #self.sld = wx.Slider(self, value=200, minValue=5, maxValue=1000, pos=(110, 505), 
        #    size=(450, -1), style=wx.SL_HORIZONTAL)
        self.sld = wx.Slider(self, value=200, minValue=5, maxValue=1000, pos=(110, 805), 
            size=(450, -1), style=wx.SL_HORIZONTAL)
        self.sld.Bind(wx.EVT_SCROLL, self.OnSliderScroll)
        
        menuBar = wx.MenuBar()
        menuBar.Append(filemenu,"&File")
        self.SetMenuBar(menuBar)
        
        #print "Made frame"
    
    def OnSliderScroll(self,event):
        #pass
        obj = event.GetEventObject()
        val = obj.GetValue()
        self.cam.set(cv.CV_CAP_PROP_EXPOSURE,val)
        #time.sleep(2)
        #print 'slide'
        
    def OnClick(self,event):
        height,width,depth = self.image.shape
        mtrx = np.zeros((10,height,width),dtype = np.uint8)
        frames = list()
        for x,frame in enumerate(self.framebuffer):
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            mtrx[x,:,:] = cv.fromarray(frame)
        rimg = cv2.fastNlMeansDenoisingMulti(mtrx,5,3)
        cv2.imwrite('out.tif', rimg)
        
class ShowCapture(wx.Panel):
    def __init__(self, parent, capture, fps=15):
        wx.Panel.__init__(self, parent)

        self.parent = parent
        self.parent.cam = cam
        
        ret, self.parent.image = self.parent.cam.read()
        image = self.parent.image
        #image = cv2.resize(self.parent.image, (0,0), fx=2, fy=2)

        height, width = self.parent.image.shape[:2]
        parent.SetSize((width*1, height*1.35))
        self.SetSize((1*width, 1*height))
        #frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        self.bmp = wx.BitmapFromBuffer(width, height, image)

        self.timer = wx.Timer(self)
        self.timer.Start(1000./fps)

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_TIMER, self.NextFrame)
        
        #filemenu= wx.Menu()
        #menuBar = wx.MenuBar()
        #menuBar.Append(filemenu,"&File")
        #self.SetMenuBar(menuBar)
        
        

    def OnPaint(self, evt):
        dc = wx.BufferedPaintDC(self)
        dc.DrawBitmap(self.bmp, 0, 0)

    def NextFrame(self, event):
        ret, self.parent.image = self.parent.cam.read()
        self.parent.framebuffer.append(copy.copy(self.parent.image))
        if ret:
            #frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            #image = cv2.resize(self.parent.image, (0,0), fx=2, fy=2)
            image = self.parent.image
            #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            self.bmp.CopyFromBuffer(image)
            self.Refresh()

#cams = VimbaCamera.getAvailableCameras(True)
#cam = VimbaCamera(cams[0]['id'])
#cam.grabStart()

class AVTCam(object):
    def __init__(self,index):
        cams = VimbaCamera.getAvailableCameras(True)
        self.camera = VimbaCamera(cams[index]['id'])
        self.camera.grabStart()
    
    def __del__(self):
        pyVimbaShutdown()
    
    def read(self):
        try:
            frame = self.xsxcamera.getImage(2000)
            ret = True
        except:
            frame = None
            ret = False
        return ret,frame
        
    def set(self,prop,value):
        if prop == cv.CV_CAP_PROP_EXPOSURE:
            self.camera.setFeature('ExposureTimeAbs',value*1000)
            #cv.CV_CAP_PROP_EXPOSURE,.50)
        
#cam = cv2.VideoCapture(0)
#cam = AVTCam(0)
#cam.set(cv.CV_CAP_PROP_FRAME_WIDTH, 320)
#cam.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 240)

try:
    from pyVimba import VimbaCamera, pyVimbaShutdown
    cam = AVTCam(0)
except ImportError:
    cam = cv2.VideoCapture(0)
    
    
app = wx.App()
aframe = PlateGui(parent=None,id=-1,title="Test Frame")
cap = ShowCapture(aframe, cam)
aframe.Show()
app.MainLoop()
#pyVimbaShutdown()
cam.release()