import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.uic import loadUi
import time
import cv2 as cv
from reconhecedor import Reconhecedor, Register,setTestImages
username = ""
Capture = cv.VideoCapture(0)

class MainWindow(QDialog):
  def __init__(self):
    super(MainWindow,self).__init__()
    loadUi("Telas/MainWindow.ui",self)
    self.pushButton.clicked.connect(self.goForward)
    self.FeedLabel = self.label
    self.Worker1 = Worker1()
    self.Worker1.start()
    self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
    self.Worker1.Success = self.Success
  def ImageUpdateSlot(self,Image):
    self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

  def Success(self):
    self.status.setText("Status: Sucesso")
    time.sleep(2)
    self.status.setText("Status: Reconhecendo")

  def goForward(self):
    widget.setCurrentIndex(widget.currentIndex()+1)
    # self.Worker1.stop()

class RegisterNameWindow(QDialog):
  def __init__(self):
    super(RegisterNameWindow,self).__init__()
    loadUi("Telas/RegisterHandWindow.ui",self)
    self.back.clicked.connect(self.goBack)
    self.forward.clicked.connect(self.goForward)

  def goBack(self):
    widget.setCurrentIndex(widget.currentIndex()-1)

  def goForward(self):
    global username
    username = self.textEdit.toPlainText()
    widget.setCurrentIndex(widget.currentIndex()+1)


class RegisterHandWindow(QDialog):
  def __init__(self):
    super(RegisterHandWindow,self).__init__()
    loadUi("Telas/RegisterHandWindow2.ui",self)
    self.Worker2 = Worker2()
    self.Worker2.start()
    self.Worker2.ImageUpdate.connect(self.ImageUpdateSlot)

  def ImageUpdateSlot(self,Image):
    self.label.setPixmap(QPixmap.fromImage(Image))

class Worker1(QThread):
  ImageUpdate = pyqtSignal(QImage)

  def run(self):
    self.ThreadActive = True
    handCountour = cv.imread("./ContourHand/hand contourn.png")

    while self.ThreadActive:
      try:
        if(widget.currentIndex()== 0):
          ret, frame= Capture.read()
          if ret:
            resized = cv.resize(frame,(500,500))
            andimage = cv.bitwise_and(handCountour,resized)

            result = Reconhecedor()
            detected,image = result.detectHand(andimage)

            Image = cv.cvtColor(image,cv.COLOR_GRAY2RGB)

            FlippedImage = cv.flip(Image, 1)
            ConvertToQtFormat = QImage(FlippedImage.data,FlippedImage.shape[1],FlippedImage.shape[0],QImage.Format_RGB888)
            Pic = ConvertToQtFormat.scaled(500,500, Qt.KeepAspectRatio)

            if(detected == True):
              self.Success()
            self.ImageUpdate.emit(Pic)
      except:
        print("Erro ao detectar")

  def stop(self):
    self.ThreadActive = False

class Worker2(QThread):
  ImageUpdate = pyqtSignal(QImage)
  def run(self):
    self.ThreadActive = True
    handCountour = cv.imread("./ContourHand/hand contourn.png")
    result = Register()
    global username

    while self.ThreadActive:
      try:
        if(widget.currentIndex() == 2):
          ret, frame= Capture.read()
          if ret:
            resized = cv.resize(frame,(500,500))
            andimage = cv.bitwise_and(handCountour,resized)
            image = result.getHand(andimage)
            Image = cv.cvtColor(image,cv.COLOR_GRAY2BGR)

            FlippedImage = cv.flip(Image, 1)
            ConvertToQtFormat = QImage(FlippedImage.data,FlippedImage.shape[1],FlippedImage.shape[0],QImage.Format_RGB888)
            Pic = ConvertToQtFormat.scaled(500,500, Qt.KeepAspectRatio)
            self.ImageUpdate.emit(Pic)
            self.sleep(1)

          if result.getCounter() == 3:
            result.createFolderUser(username)
            result.setImagesToRegister(username)
            result.reset()
            setTestImages()

            widget.setCurrentIndex(0)
      except:
        print("error")

  def stop(self):
    self.ThreadActive = False
    self.stop()


if __name__ == "__main__":
  App = QApplication(sys.argv)
  widget= QStackedWidget()
  main = MainWindow()
  registerName = RegisterNameWindow()
  registerHand = RegisterHandWindow()
  widget.setFixedHeight(600)
  widget.setFixedWidth(540)
  widget.addWidget(main)
  widget.addWidget(registerName)
  widget.addWidget(registerHand)
  widget.show()
  sys.exit(App.exec())
