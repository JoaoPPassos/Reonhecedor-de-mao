import cv2 as cv
import argparse
from PIL import Image

vid = cv.VideoCapture(0)

right = cv.CascadeClassifier("template.xml")

template = cv.imread("./maos_reconheciveis/mao 1.png",0)

max_value = 255
max_type = 4
max_binary_value = 255
trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
window_name = 'Threshold Demo'

threshold_type = ""
threshold_value = ""

cv.namedWindow(window_name)

def Threshold_Demo(val):
    #0: Binary
    #1: Binary Inverted
    #2: Threshold Truncated
    #3: Threshold to Zero
    #4: Threshold to Zero Inverted
    global threshold_type 
    global threshold_value

    threshold_type = cv.getTrackbarPos(trackbar_type, window_name)
    threshold_value = cv.getTrackbarPos(trackbar_value, window_name)
    
def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    
    #0: Binary
    #1: Binary Inverted
    #2: Threshold Truncated
    #3: Threshold to Zero
    #4: Threshold to Zero Inverted
    
    #-- Detect faces
    
    righthand = right.detectMultiScale(frame_gray,1.2,7)
    # cv.rectangle(frame_gray, ())
    # cv.rectangle(img,(384,0),(510,128),(0,255,0),3)
    
    if righthand == ():
      print("RightHand not available")
    else:
      global threshold_type 
      global threshold_value
      
      for(x,y,w,h) in righthand:
        # cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        cropimage = frame_gray[y:y+h,x:x+w]
        # frame_HSV = cv.cvtColor(cropimage, cv.COLOR_BGR2HSV)
        # frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
   
        _, dst = cv.threshold(cropimage, threshold_value, max_binary_value, threshold_type )
        
        handChecker(dst)
        cv.imshow("hand",dst)
        cv.resizeWindow("hand",200,400)
    
  
def handChecker(frame):
  result = cv.matchTemplate(frame,template,cv.TM_CCOEFF)
  
  print(result)
  
cv.createTrackbar(trackbar_type, window_name , 3, max_type,Threshold_Demo)
cv.createTrackbar(trackbar_value, window_name , 1, max_value,Threshold_Demo)


while (vid.isOpened()):
  ret,frame = vid.read()

  detectAndDisplay(frame)
  
  
  cv.imshow(window_name, frame)

  if cv.waitKey(1) == ord('q'):
    cv.destroyAllWindows()
    break
