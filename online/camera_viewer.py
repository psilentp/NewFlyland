import wx
import cv2
from pyVimba import VimbaCamera, pyVimbaShutdown

class PlateGui(wx.Frame):

    def __init__(self, *args , **kwds):
        self.frame = wx.Frame.__init__(self,*args, **kwds)
        filemenu= wx.Menu()
        menuBar = wx.MenuBar()
        menuBar.Append(filemenu,"&File")
        self.SetMenuBar(menuBar)
        #print "Made frame"
        
class ShowCapture(wx.Panel):
    def __init__(self, parent, capture, fps=15):
        wx.Panel.__init__(self, parent)

        self.cam = cam
        frame = self.cam.getImage(2000)
        #ret, frame = self.capture.read()

        height, width = frame.shape[:2]
        parent.SetSize((width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.bmp = wx.BitmapFromBuffer(width, height, frame)

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
        #ret, frame = self.capture.read()
        frame = self.cam.getImage(2000)
        ret = True
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            self.bmp.CopyFromBuffer(frame)
            self.Refresh()

cams = VimbaCamera.getAvailableCameras(True)
cam = VimbaCamera(cams[0]['id'])
cam.grabStart()

#capture = cv2.VideoCapture(0)
#capture.set(cv.CV_CAP_PROP_FRAME_WIDTH, 320)
#capture.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 240)

app = wx.App()
aframe = PlateGui(parent=None,id=-1,title="Test Frame")
panel = wx.Panel(parent=aframe)
cap = ShowCapture(panel, cam)
aframe.Show()
#panel.Show()
app.MainLoop()