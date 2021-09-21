import sys
import os
import cv2
from SimpleHigherHRNet import SimpleHigherHRNet
import numpy as np
import math

model = None
device = 'cpu'
class_map = {}
return_format = 'img_with_pose'  #img_with_pose / only_pose

def get_support_models():
    return ['higher_hrnet_w32_512', ]

def create_model(model_name='higher_hrnet_w32_512', dev='cpu'):
    global model
    global device
    model = None
    device = dev

    if model_name == 'higher_hrnet_w32_512':
        channels = 32 #HRNet: 32, 48
        joints_num = 17 #COCO: 17, CrowdPose: 14
        pre_train = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pose_higher_hrnet_w32_512.pth')
        model = SimpleHigherHRNet(channels, joints_num, pre_train, device=dev)
    else:
        print('unsupport model_name')
        exit(0)

def calulate_pose_similiar(kp_map1, kp_map2):
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

    return angle_map


def _phase_pose(kps):
    def _calculate_angle(point1, point2):
        new_point = [point2[0] - point1[0], point2[1] - point1[1]]
        if new_point[0] == 0:
            new_point[0] = 1
        if new_point[1] == 0:
            new_point[1] = 1

        angle = math.atan( new_point[1] / new_point[0] )

        # first
        if new_point[0] > 0 and new_point[1] > 0:
            angle = angle
        # second
        elif new_point[0] < 0 and new_point[1] > 0:
            angle = 3.1415926 + angle
        # third
        elif new_point[0] < 0 and new_point[1] < 0:
            angle = 3.1415926 + angle
        # forth
        else:
            angle = 2 * 3.1415926 + angle
        return angle * 180 / 3.1415926


    kps = kps[:,[1,0]] #[y,x,score] -> [x,y]
    #kps[:, 0] = kps[:, 0] #*x_scale
    #kps[:, 1] = kps[:, 1] #*x_scale
    kps = kps.astype(np.int32)
    keypoint_map = {'nose': kps[0].tolist(),
                    'left_shoulder': kps[5].tolist(),
                    'right_shoulder': kps[6].tolist(),
                    'center_shoulder': ((kps[5] + kps[6]) / 2).astype(np.int32).tolist(),
                    'left_elbow': kps[7].tolist(),
                    'right_elbow': kps[8].tolist(),
                    'left_wrist': kps[9].tolist(),
                    'right_wrist': kps[10].tolist(),
                    'left_hip': kps[11].tolist(),
                    'right_hip': kps[12].tolist(),
                    'center_hip': ((kps[11] + kps[12]) / 2).astype(np.int32).tolist(),
                    'left_knee': kps[13].tolist(),
                    'right_knee': kps[14].tolist(),
                    'left_ankle': kps[15].tolist(),
                    'right_ankle': kps[16].tolist(),}
    return keypoint_map

def _draw_pose(img, pose_map):
    thickness = 2
    linetype = 8
    color = (0,0,255)
    cv2.line(img, tuple(pose_map['nose']), tuple(pose_map['center_shoulder']), color, thickness, linetype)
    cv2.line(img, tuple(pose_map['center_shoulder']), tuple(pose_map['center_hip']), color, thickness, linetype)
    cv2.line(img, tuple(pose_map['left_shoulder']), tuple(pose_map['right_shoulder']), color, thickness, linetype)
    cv2.line(img, tuple(pose_map['left_hip']), tuple(pose_map['right_hip']), color, thickness, linetype)
    cv2.line(img, tuple(pose_map['left_shoulder']), tuple(pose_map['left_elbow']), color, thickness, linetype)
    cv2.line(img, tuple(pose_map['left_elbow']), tuple(pose_map['left_wrist']), color, thickness, linetype)
    cv2.line(img, tuple(pose_map['right_shoulder']), tuple(pose_map['right_elbow']), color, thickness, linetype)
    cv2.line(img, tuple(pose_map['right_elbow']), tuple(pose_map['right_wrist']), color, thickness, linetype)
    cv2.line(img, tuple(pose_map['left_hip']), tuple(pose_map['left_knee']), color, thickness, linetype)
    cv2.line(img, tuple(pose_map['left_knee']), tuple(pose_map['left_ankle']), color, thickness, linetype)
    cv2.line(img, tuple(pose_map['right_hip']), tuple(pose_map['right_knee']), color, thickness, linetype)
    cv2.line(img, tuple(pose_map['right_knee']), tuple(pose_map['right_ankle']), color, thickness, linetype)



def inference(img_array, return_format='img_with_pose'):
    global model
    global class_map
    map_result = {'type':'img'}
    
    joints = model.predict(img_array)

    if return_format == 'only_pose':
        img_array = np.zeros_like(img_array).as_type(np.uint8)

    phase_person_num = 1
    pose_map = None
    for i, one_person_kps in enumerate(joints):
        if i >= phase_person_num:
            break
        pose_map = _phase_pose(one_person_kps)
        _draw_pose(img_array, pose_map)

        
    map_result['result'] = img_array
    map_result['pose_map'] = pose_map
    
    return map_result


if __name__ == '__main__':
    create_model(model_name='higher_hrnet_w32_512', dev='cuda')
    
    image1 = cv2.imread("people.jpg")
    results1 = inference(image1, return_format='img_with_pose', handle=None)
    
    image2 = cv2.imread("people3.jpg")
    results2 = inference(image2, return_format='img_with_pose', handle=None)
    print('img1:', results1['pose_map'])
    print('img2:', results2['pose_map'])

    similier = calulate_pose_similiar(results1['pose_map'], results2['pose_map'])
    print(similier)