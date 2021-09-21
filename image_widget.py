# -*- coding: utf-8 -*-
import sys
import os
if hasattr(sys, 'frozen'):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import copy
import xml.etree.cElementTree as et
import os
import cv2
import math
from PIL import Image
from alg_pytorch import AlgPytorch 

# ui配置文件
cUi, cBase = uic.loadUiType("image_widget.ui")

# 主界面
class ImageWidget(QWidget, cUi):
    def __init__(self, id=0): #, main_widget=None):
        # 设置UI
        QMainWindow.__init__(self)
        cUi.__init__(self)
        self.setupUi(self)

        self.comboBoxCamera.addItem('0')
        self.comboBoxCamera.addItem('1')
        self.comboBoxCamera.addItem('2')
        
        self.timer = QTimer()
        self.video_cap = None
        self.camera_cap = None
        
        self.qpixmap = None
        self.cAlg = None
        self.infer = None
        self.class_map  = None
        self.alg_time = None

        self.id = id
        
        self.color_list = [QColor(255,0,0),
                      QColor(0,255,0),
                      QColor(0,0,255),
                      QColor(0,255,255),
                      QColor(255,0,255),
                      QColor(8,46,84),
                      QColor(199,97,20),
                      QColor(255,227,132),
                      QColor(255,255,0),
                      QColor(128,138,135)]

        if self.id == 1:
            self.btnVideo.setVisible(False)
            self.btnCamera.setVisible(False)
            self.btnStop.setVisible(False)
            self.comboBoxCamera.setVisible(False)
        
    @pyqtSlot()
    def on_btnPhoto_clicked(self):
        print('on_btnPhoto_clicked')
        img_path = QFileDialog.getOpenFileName(self,  "选取图片", "./", "Images (*.jpg);;Images (*.png)") 
        img_path = img_path[0]
        if img_path != '':
            self.slot_photo_frame(img_path)
    
    @pyqtSlot()
    def on_btnVideo_clicked(self):
        print('on_btnVideo_clicked')
        video_path = QFileDialog.getOpenFileName(self,  "选取视频", "./", "Videos (*.mp4);;Images (*.3gp)") 
        video_path = video_path[0]
        if video_path != '':
            self.video_cap = cv2.VideoCapture(video_path)
            self.timer.start()
            self.timer.setInterval(int(1000 / float(30.0)))
            self.timer.timeout.connect(self.slot_video_frame)
                    
    @pyqtSlot()    
    def on_btnCamera_clicked(self):
        print('on_btnCamera_clicked')
        if self.camera_cap is None:
            self.camera_cap = cv2.VideoCapture(int(0))
            self.timer.start()
            self.timer.setInterval(int(1000 / float(30.0)))
            self.timer.timeout.connect(self.slot_camera_frame)
        else:
            self.camera_cap.release()
            self.camera_cap = None
            self.timer.stop()
                    
    @pyqtSlot()    
    def on_btnStop_clicked(self):
        self.stop_all()
            
    def slot_photo_frame(self, photo_path):          
        img = cv2.imread(photo_path)        
        self.cAlg.add_img((img, self.id))
                       
    def slot_camera_frame(self):
        if self.camera_cap is not None:
            # get a frame
            ret, img = self.camera_cap.read()
            if ret is False:
                self.stop_all()
                return
            self.cAlg.add_img((img, self.id))
        
    def slot_video_frame(self):
        if self.video_cap is not None:
            ret, img = self.video_cap.read()
            if ret is False:
                self.stop_all()
                return 
            self.cAlg.add_img((img,self.id))
                      
    def slot_alg_result(self, img, result, time_spend):
        if result['type'] == 'info':
            print(result['result'])
            return
        elif result['type'] == 'img':
            img = result['result']
            self.infer = None
        else:
            print('error type')
                
        height, width, bytesPerComponent = img.shape
        bytesPerLine = bytesPerComponent * width
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        image = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.qpixmap = QPixmap.fromImage(image) 
        self.alg_time = time_spend
        self.update()
    
    
    def stop_all(self):
        self.timer.stop()
        self.qpixmap = None
        if self.camera_cap is not None:
            self.camera_cap.release()
            self.camera_cap = None
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
            
    def set_alg_handle(self, handle):
        self.cAlg = handle
        
        
    def draw_image(self, painter):
        pen = QPen()
        font = QFont("Microsoft YaHei")
        if self.qpixmap is not None:
            painter.drawPixmap(QtCore.QRect(0, 0, self.width(), self.height()), self.qpixmap)
            pen.setColor(self.getColor(0))
            #painter.setPen(pen)
            #pointsize = font.pointSize()
            #font.setPixelSize(pointsize*180/72)
            #painter.setFont(font)
            #painter.drawText(10, 30, 'time=%.4f seconds fps=%.4f' % (self.alg_time, 1 / self.alg_time))
        else:
            pen.setColor(QColor(0, 0, 0))
            pen.setWidth(4)
            painter.setPen(pen)
            painter.drawRect(0, 0, self.width(), self.height())
        
    def draw_extra_info(self, painter):
        x0 = self.width() * 0.05 
        y0 = self.height() * 0.05
        x1 = self.width() * 0.4
        y1 = self.height() * 0.4
        rect = QRect(x0,y0,x1,y1)
        painter.setPen(QPen(Qt.white, 5 ,Qt.SolidLine))
        painter.drawRect(rect)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        self.draw_image(painter)
        #self.draw_extra_info(painter)
        
    def getColor(self, index):
        return self.color_list[index % len(self.color_list)]
        
if __name__ == "__main__":
    cApp = QApplication(sys.argv)
    cImageWidget = ImageWidget()
    cImageWidget.show()
    sys.exit(cApp.exec_())