from time import sleep
import os
import numpy as np
import imagehash
import cv2 as cv
from PIL import Image
from scipy.spatial import distance

vid = cv.VideoCapture(0)

right = cv.CascadeClassifier("template.xml")


handCountour = cv.imread("./ContourHand/hand contourn.png",0)
th,tw,c = (0,0,0)
max_value = 255
max_type = 4
max_binary_value = 255
trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
window_name = 'Threshold Demo'

test_images = []
images_to_regist = []
threshold_type = ""
threshold_value = ""
    
def setTestImages():
  caminhos = os.listdir("./maos_reconheciveis")
  user_hands = []
  for url in caminhos:
    for i in range(0 ,3):
      user_hands.append("./maos_reconheciveis/{}/img_{}.png".format(url,i))
  test_images.append(user_hands)
    

def detectHand(frame):
  # frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  global threshold_type 
  global threshold_value
  frame_gray = cv.equalizeHist(frame)
  
  righthand = right.detectMultiScale(frame_gray,1.2,7)

  return righthand
  
class Register():
  def __init__(self):
    self.handList = []
    self.img_counter = 0
    setTestImages()
    
  def reset(self):
    self.handList =[]
    self.img_counter = 0
    
  def getHand(self,image):
    value,showimage, detected = registerHand(image)
    
    if(value == 1):
      self.handList.append(detected)
      self.img_counter+=1
    return cv.cvtColor(showimage,cv.COLOR_BGR2GRAY)
  
  def getCounter(self):
    return self.img_counter
  
  def createFolderUser(self,username):
    os.mkdir("./maos_reconheciveis/{}".format(username))

  def setImagesToRegister(self,username):
    for index,img in enumerate(self.handList): 
      url ="./maos_reconheciveis/{}/img_{}.png".format(username,index)
      cv.imwrite(url,img)
    setTestImages()

def registerHand(andimage):
  grayframe = cv.cvtColor(andimage,cv.COLOR_BGR2GRAY)
  grayframe = cv.equalizeHist(grayframe)
  
  righthand = detectHand(grayframe)

  if(isinstance(righthand, np.ndarray)):
    for (x,y,w,h) in righthand:
      cv.rectangle(andimage,(x,y),(x+w,y+h),(0,255,0),3)
      cropimage =  cv.equalizeHist(grayframe[y:y+h,x:x+w])
  
      _, dst = cv.threshold(cropimage, 120, max_binary_value, 2 )
      
    return 1,andimage,dst
  return 0,andimage,grayframe

  

  
def detectAndDisplay(andframe):
    global threshold_type 
    global threshold_value
    grayframe = cv.cvtColor(andframe,cv.COLOR_BGR2GRAY)
    grayframe = cv.equalizeHist(grayframe)
    righthand = right.detectMultiScale(grayframe,1.2,7)
    detected = False
    if righthand == ():
      x= 0
    else:
      for(x,y,w,h) in righthand:
        cv.rectangle(grayframe,(x,y),(x+w,y+h),(0,255,0),3)
        cropimage = cv.equalizeHist(grayframe[y:y+h,x:x+w])

        _, dst = cv.threshold(cropimage, 120, max_binary_value, 2 )

        detected = handChecker(dst)
    return detected,grayframe
 
def handChecker(frame):
  frame = Image.fromarray(frame)
  frame_hash = imagehash.average_hash(frame)
  for images in test_images:
    for url in images:
      template = cv.imread(url,0)
      template = Image.fromarray(template)
      template_hash = imagehash.average_hash(template)

      result = frame_hash - template_hash
      if(result <= 15): 
        return True
  return False

class Reconhecedor():
  def detectHand(self,image):
    detected,result = detectAndDisplay(image)
    return detected,result
