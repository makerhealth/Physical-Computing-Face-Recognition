from facedetector import FaceDetector
from picamera.array import PiRGBArray
from picamera import PiCamera
import imutils
import cv2
import time

#  DEFINE CONSTANTS FOR CAMERA ON RASPBERRY PI
framerate = 16
camera_warmup_time = 2.5


#  Set up Raspberry Pi 2 serial camera and let it warm up
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = framerate
rawCapture = PiRGBArray(camera, size = (640, 480))
time.sleep(camera_warmup_time)

#  Instantiate FaceDetector object and pass it HarrCascade XML face classifier
fd = FaceDetector("haarcascade_frontalface_default.xml")

#  Begin Capturing frames from camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    image = imutils.resize(image, width = 300)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceRects = fd.detect(gray, scaleFactor = 1.1, minNeighbors = 5,
                          minSize = (30, 30))
    frameClone = image.copy()
    for(fX, fY, fW, fH) in faceRects:
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                     (0, 255, 0), 2)
        
    cv2.imshow("Face", frameClone)
    if cv2.waitKey(1) & 0xFF == ord("q"):
           break
    
    rawCapture.truncate(0)






           
