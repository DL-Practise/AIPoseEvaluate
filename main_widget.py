# -*- coding: utf-8 -*-
import sys
import os
if hasattr(sys, 'frozen'):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from image_widget import ImageWidget
from alg_pytorch import AlgPytorch 
from model_manager import ModelManager
from common_utils import get_api_from_model
import math

# ui配置文件
cUi, cBase = uic.loadUiType("main_widget.ui")

# 主界面
class MainWidget(QWidget, cUi):
    def __init__(self): #, main_widget=None):
        # 设置UI
        QMainWindow.__init__(self)
        cUi.__init__(self)
        self.setupUi(self)
        
        
        self.cImageWidget1 = ImageWidget(id=1)
        self.cImageWidget2 = ImageWidget(id=2)
        self.cAlg = AlgPytorch(self.alg_result_cb)
        self.cModelManager = ModelManager(self.alg_select_cb)
        self.cImageWidget1.set_alg_handle(self.cAlg)     
        self.cImageWidget2.set_alg_handle(self.cAlg)    

        self.grid_image1.addWidget(self.cImageWidget1)
        self.grid_image2.addWidget(self.cImageWidget2)
        self.grid_model_manager.addWidget(self.cModelManager)
        
        self.std_pose = None
        self.compare_pose = None
        self.score = 0

        self.show_score(0)


    def show_score(self, score):
        if score == 100:
            bai_img = './pics/1.jpg'
            shi_img = './pics/0.jpg'
            ge_img = './pics/0.jpg'
        else:
            bai_img = './pics/0.jpg'
            shi_img = './pics/%d.jpg'%(score / 10)
            ge_img = './pics/%d.jpg'%(score % 10)
        fen_img = './pics/fen.jpg'
        self.label_bai.setScaledContents (True)
        self.label_bai.setPixmap(QPixmap(bai_img).scaled(self.label_bai.size()))
        self.label_shi.setScaledContents (True)
        self.label_shi.setPixmap(QPixmap(shi_img).scaled(self.label_shi.size()))
        self.label_ge.setScaledContents (True)
        self.label_ge.setPixmap(QPixmap(ge_img).scaled(self.label_ge.size()))
        self.label_fen.setScaledContents (True)
        self.label_fen.setPixmap(QPixmap(fen_img).scaled(self.label_fen.size()))

    def alg_select_cb(self, model_name, model_path, device):
        print(model_name, model_path)
        self.cAlg.create_model(model_dir=model_path, model_name=model_name, device=device)
        self.logEdit.append('select alg: ' + model_name + ', device:' 
            + device + ', please wait for creating model!!!')

    def alg_result_cb(self, img, id, result, time_spend):
        if result['type'] == 'info':
            self.logEdit.append(result['result'])
        else:
            if id == 1:
                self.std_pose = result['pose_map']
                self.cImageWidget1.slot_alg_result(img, result, time_spend)
            else:
                self.compare_pose = result['pose_map']
                self.cImageWidget2.slot_alg_result(img, result, time_spend)
                if self.std_pose is None or self.compare_pose is None:
                    self.score = 0
                    self.show_score(self.score)
                else:
                    angles = self.calulate_pose_similiar(self.std_pose, self.compare_pose)
                    self.score = angles['score']
                    print(self.score)
                    self.show_score(self.score)

    def paintEvent(self, event):
        self.show_score(self.score)

    def closeEvent(self, event):        
        reply = QMessageBox.question(self, 'Message',"Are you sure to quit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
            self.cImageWidget1.stop_all()
            self.cImageWidget2.stop_all()
            self.cAlg.quit()
        else:
            event.ignore()

    def calulate_pose_similiar(self, kp_map1, kp_map2):
        def _calculate_angle(line1_p1, line1_p2, line2_p1, line2_p2):
            dx1 = line1_p1[0] - line1_p2[0]
            dy1 = line1_p1[1] - line1_p2[1]
            dx2 = line2_p1[0] - line2_p2[0]
            dy2 = line2_p1[1] - line2_p2[1]
            angle1 = math.atan2(dy1, dx1)
            angle1 = int(angle1 * 180 / math.pi)
            angle2 = math.atan2(dy2, dx2)
            angle2 = int(angle2 * 180 / math.pi)
            if angle1 * angle2 >= 0:
                insideAngle = abs(angle1 - angle2)
            else:
                insideAngle = abs(angle1) + abs(angle2)
                if insideAngle > 180:
                    insideAngle = 360 - insideAngle
            insideAngle = insideAngle % 180
            return insideAngle

        angle_map = {'angle_neck': _calculate_angle(kp_map1['nose'], kp_map1['center_shoulder'], kp_map2['nose'], kp_map2['center_shoulder']),
                    'angle_shoulder': _calculate_angle(kp_map1['left_shoulder'], kp_map1['right_shoulder'], kp_map2['left_shoulder'], kp_map2['right_shoulder']),
                    'angle_body': _calculate_angle(kp_map1['center_shoulder'], kp_map1['center_hip'], kp_map2['center_shoulder'], kp_map2['center_hip']),
                    'angle_hip': _calculate_angle(kp_map1['left_hip'], kp_map1['right_hip'], kp_map2['left_hip'], kp_map2['right_hip']),
                    'angle_left_arm1': _calculate_angle(kp_map1['left_shoulder'], kp_map1['left_elbow'], kp_map2['left_shoulder'], kp_map2['left_elbow']),
                    'angle_left_arm2': _calculate_angle(kp_map1['left_elbow'], kp_map1['left_wrist'], kp_map2['left_elbow'], kp_map2['left_wrist']),
                    'angle_right_arm1': _calculate_angle(kp_map1['right_shoulder'], kp_map1['right_elbow'], kp_map2['right_shoulder'], kp_map2['right_elbow']),
                    'angle_right_arm2': _calculate_angle(kp_map1['right_elbow'], kp_map1['right_wrist'], kp_map2['right_elbow'], kp_map2['right_wrist']),
                    'angle_left_leg1': _calculate_angle(kp_map1['left_hip'], kp_map1['left_knee'], kp_map2['left_hip'], kp_map2['left_knee']),
                    'angle_left_leg2': _calculate_angle(kp_map1['left_knee'], kp_map1['left_ankle'], kp_map2['left_knee'], kp_map2['left_ankle']),
                    'angle_right_leg1': _calculate_angle(kp_map1['right_hip'], kp_map1['right_knee'], kp_map2['right_hip'], kp_map2['right_knee']),
                    'angle_right_leg2': _calculate_angle(kp_map1['right_knee'], kp_map1['right_ankle'], kp_map2['right_knee'], kp_map2['right_ankle']),}
        total_angle = 0
        for key in angle_map.keys():
            total_angle += angle_map[key]
        
        angle_map['total'] = total_angle
        ratio = (2160 - total_angle) / 2160
        ratio = ratio * ratio
        angle_map['score'] = int(ratio * 100)

        return angle_map
        
if __name__ == "__main__":
    cApp = QApplication(sys.argv)
    cMainWidget = MainWidget()
    cMainWidget.show()
    sys.exit(cApp.exec_())