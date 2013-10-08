import cv2
from pyVimba import VimbaCamera, pyVimbaShutdown

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

cams = VimbaCamera.getAvailableCameras(True)
cam = VimbaCamera(cams[0]['id'])
cam.grabStart()


if cam.getFeature("AcquisitionStart"): # try to get the first frame
    frame = cam.getImage(2000)
    rval = True
else:
    rval = False

while rval:
    cv2.imshow("preview", frame)
    frame = cam.getImage(2000)
    #frame = cv2.Laplacian(frame,5)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
    
cv2.destroyWindow('preview')
pyVimbaShutdown()