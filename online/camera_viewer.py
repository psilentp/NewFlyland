import wx
import cv2
from cv2 import cv
import pylab as plb
import time
#from pyVimba import VimbaCamera, pyVimbaShutdown

class PlateGui(wx.Frame):

    def __init__(self, *args , **kwds):
        self.frame = wx.Frame.__init__(self,*args, **kwds)
        filemenu= wx.Menu()
        #self.toolbar = self.CreateToolBar(style = wx.TB_BOTTOM | wx.TB_FLAT)
        #qtool = self.toolbar.AddLabelTool(wx.ID_ANY, 'Quit',None)
        #self.toolbar.AddButton(self, wx.ID_OK, "Ok")
        #self.toolbar.Realize()
        okButton = wx.Button(self, wx.ID_OK, "Save",pos=(0, 500))
        self.Bind(wx.EVT_BUTTON, self.OnClick, okButton)
        
        self.sld = wx.Slider(self, value=200, minValue=150, maxValue=500, pos=(110, 505), 
            size=(530, -1), style=wx.SL_HORIZONTAL)
        self.sld.Bind(wx.EVT_SCROLL, self.OnSliderScroll)
        
        menuBar = wx.MenuBar()
        menuBar.Append(filemenu,"&File")
        self.SetMenuBar(menuBar)
        
        #print "Made frame"
    
    def OnSliderScroll(self,event):
        self.cam.set(cv.CV_CAP_PROP_EXPOSURE,.50)
        #time.sleep(2)
        #print 'slide'
        
    def OnClick(self,event):
        plb.imsave('out.tiff',self.image)
        
class ShowCapture(wx.Panel):
    def __init__(self, parent, capture, fps=15):
        wx.Panel.__init__(self, parent)

        self.parent = parent
        self.parent.cam = cam
        
        #frame = self.cam.getImage(2000)
        ret, self.parent.image = self.parent.cam.read()
        image = cv2.resize(self.parent.image, (0,0), fx=2, fy=2)

        height, width = self.parent.image.shape[:2]
        parent.SetSize((width*2, height*2.35))
        self.SetSize((2*width, 2*height))
        #frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        self.bmp = wx.BitmapFromBuffer(2*width, 2*height, image)

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
        #frame = self.cam.getImage(2000)
        #ret = True
        if ret:
            #frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            image = cv2.resize(self.parent.image, (0,0), fx=2, fy=2)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            self.bmp.CopyFromBuffer(image)
            self.Refresh()

#cams = VimbaCamera.getAvailableCameras(True)
#cam = VimbaCamera(cams[0]['id'])
#cam.grabStart()

cam = cv2.VideoCapture(0)
cam.set(cv.CV_CAP_PROP_FRAME_WIDTH, 320)
cam.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 240)

app = wx.App()
aframe = PlateGui(parent=None,id=-1,title="Test Frame")
cap = ShowCapture(aframe, cam)
aframe.Show()
app.MainLoop()
cam.release()