import cv2 as cv
import argparse

from cv2 import resizeWindow

vid = cv.VideoCapture(0)

right = cv.CascadeClassifier("template.xml")

template = cv.imread("./maos_reconheciveis/mao 1.png",0)
handCountour = cv.imread("./ContourHand/hand contourn.png",0)
th,tw,*_ = template.shape

max_value = 255
max_type = 4
max_binary_value = 255
trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
window_name = 'Threshold Demo'

test_images = []

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

def setTestImages():
  for i in range(1 ,6):
    test_images.append("imagem_{}.jpeg".format(i))
    
setTestImages()

def detectAndDisplay(frame):
    # frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    global threshold_type 
    global threshold_value
    frame_gray = cv.equalizeHist(frame)
    
    #0: Binary
    #1: Binary Inverted
    #2: Threshold Truncated
    #3: Threshold to Zero
    #4: Threshold to Zero Inverted
    
    #-- Detect faces
    
    righthand = right.detectMultiScale(frame_gray,1.2,7)
    # cv.rectangle(frame_gray, ())
    # cv.rectangle(img,(384,0),(510,128),(0,255,0),3)
    _, dst = cv.threshold(frame_gray, threshold_value, max_binary_value, threshold_type )
    cv.imshow("threshold",dst)
    
    if righthand == ():
      # print("RightHand not available")
      x = 0
    else:
      
      
      for(x,y,w,h) in righthand:
        # cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        cropimage = frame_gray[y:y+th,x:x+tw]
        # frame_HSV = cv.cvtColor(cropimage, cv.COLOR_BGR2HSV)
        # frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
   
        _, dst = cv.threshold(cropimage, threshold_value, max_binary_value, threshold_type )
        
        # test = cv.Canny(cropimage,150,180)
        handChecker(dst)
        cv.imshow("hand",dst)
        cv.resizeWindow("hand",200,400)
    

def checkHands():
  for url in test_images:
    image = cv.imread("Maos/{}".format(url))
    
    frame_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    
    righthand = right.detectMultiScale(frame_gray,1.2,7)

    if righthand == ():
      # print("RightHand not available")
      x = 0
    else:
      global threshold_type 
      global threshold_value
      for(x,y,w,h) in righthand:
        # cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        cropimage = frame_gray[y:y+th,x:x+tw]
        # frame_HSV = cv.cvtColor(cropimage, cv.COLOR_BGR2HSV)
        # frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
   
        _, dst = cv.threshold(cropimage, threshold_value, max_binary_value, threshold_type )
        
        # test = cv.Canny(cropimage,150,180)
        cv.imshow("hand_test",dst)
        handChecker(dst)
        # cv.resizeWindow("hand_test",200,400)
        
def handChecker(frame):
  result = cv.matchTemplate(frame,template,cv.TM_CCORR_NORMED)

  if(result[0][0] > 0.58):
      print("result ===>",result[0])

  
cv.createTrackbar(trackbar_type, window_name , 1, max_type,Threshold_Demo)
cv.createTrackbar(trackbar_value, window_name , 27, max_value,Threshold_Demo)


#typo 1 valor 27

while (vid.isOpened()):
  ret,frame = vid.read()

  # checkHands()
  resized_frame = cv.cvtColor(cv.resize(frame,(500,500)), cv.COLOR_BGR2GRAY)
  # handCountourGrey = cv.cvtColor(handCountour,cv.COLOR_BGR2GRAY)
  
  andimage = cv.bitwise_and( handCountour,resized_frame)
  
  detectAndDisplay(resized_frame)

  cv.imshow("hand 2",andimage)
  cv.imshow(window_name, frame)

  if cv.waitKey(1) == ord('q'):
    cv.destroyAllWindows()
    break
