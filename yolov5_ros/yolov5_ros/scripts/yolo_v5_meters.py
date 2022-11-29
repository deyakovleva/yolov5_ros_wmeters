#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import torch
import rospy
import numpy as np
import ros_numpy
from std_msgs.msg import Header, String
from sensor_msgs.msg import Image
from yolov5_ros_msgs.msg import BoundingBox, BoundingBoxes
from yolov5_ros_msgs.srv import meter_response_crop, meter_response_cropResponse


class Yolo_Dect:

    flag_for_cropping = False
    im_rate = 0

    def __init__(self):

        # load parameters
        yolov5_path = rospy.get_param('/yolov5_path', '/home/itmo/yolov5_ws/src/yolov5_ros_wmeters/yolov5_ros/yolov5_ros/yolov5')

        weight_path = rospy.get_param('~weight_path', '/home/itmo/yolov5_ws/src/yolov5_ros_wmeters/yolov5_ros/yolov5_ros/weights/meters_weights.pt')
        image_topic = rospy.get_param(
            '~image_topic', '/camera/color/image_raw')
        pub_topic = rospy.get_param('~pub_topic', '/yolov5/BoundingBoxes')
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_color_frame')
        conf = rospy.get_param('~conf', '0.5')

        # load local repository(YoloV5:v6.0)
        self.model = torch.hub.load(yolov5_path, 'custom',
                                    path=weight_path, source='local')

        # which device will be used
        if (rospy.get_param('/use_cpu', 'false')):
            self.model.cpu()
        else:
            self.model.cuda()

        self.model.conf = conf
        self.color_image = Image()
        self.depth_image = Image()
        self.getImageStatus = False

        # Load class color
        self.classes_colors = {}

        # image subscribe
        self.color_sub = rospy.Subscriber(image_topic, Image, self.image_callback,
                                          queue_size=1)

        # output publishers
        self.position_pub = rospy.Publisher(
            pub_topic,  BoundingBoxes, queue_size=1)

        self.image_pub = rospy.Publisher(
            '/yolov5/detection_image',  Image, queue_size=1)

        responce_service = rospy.Service('/response_meter', meter_response_crop, self.response_srv)

        # if no image messages
        while (not self.getImageStatus) :
            rospy.loginfo("Waiting for image.")
            rospy.sleep(2)

    def response_srv(self,request):
        self.flag_for_cropping = True
        return meter_response_cropResponse(success = True)

    def image_callback(self, image):

        if abs((self.im_rate - image.header.seq))>=30:

            self.getImageStatus = True
            #self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
            self.color_image = ros_numpy.numpify(image)
            self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)

            results = self.model(self.color_image)
            # xmin    ymin    xmax   ymax  confidence  class    name

            boxs = results.pandas().xyxy[0].sort_values(by='confidence').values
        
            self.dectshow(self.color_image, boxs, image.height, image.width)
            self.im_rate = image.header.seq

    def dectshow(self, org_img, boxs, height, width):

        count = 0
        for i in boxs:
            count += 1

        for box in boxs:
            boundingBox = BoundingBox()
            boundingBox.probability =np.float64(box[4])
            boundingBox.xmin = np.int64(box[0])
            boundingBox.ymin = np.int64(box[1])
            boundingBox.xmax = np.int64(box[2])
            boundingBox.ymax = np.int64(box[3])
            boundingBox.num = np.int16(count)
            boundingBox.Class = box[-1]
            
            if box[-1] in self.classes_colors.keys():
                color = self.classes_colors[box[-1]]
            else:
                color = np.random.randint(0, 183, 3)
                self.classes_colors[box[-1]] = color

            cv2.rectangle(org_img, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), (int(color[0]),int(color[1]), int(color[2])), 1)

            if box[1] < 20:
                text_pos_y = box[1] + 30
            else:
                text_pos_y = box[1] - 10
                
            cv2.putText(org_img, box[-1],
                        (int(box[0]), int(text_pos_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(org_img, str(np.round(box[4], 2)),
                        (int(box[0]), int(text_pos_y)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            

        if self.flag_for_cropping:
            cropped_img = org_img[boundingBox.ymin:boundingBox.ymax, boundingBox.xmin:boundingBox.xmax]
            # cv2.imwrite('/home/itmo/yolov5_ws/src/yolov5_ros_wmeters/yolov5_ros/yolov5_ros/media/meter_cropped.jpg',cropped_img)
            cv2.imwrite('/home/diana/yolov5_ros_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/media/meter_cropped.jpg',cropped_img)
            self.flag_for_cropping = False

        if (count != 0):
            rospy.loginfo("Water meter is detected")

        self.publish_image(org_img, height, width)

    def publish_image(self, imgdata, height, width):
        #image_temp = Image()
        image_temp = ros_numpy.msgify(Image, imgdata, encoding='bgr8')
        header = Header(stamp=rospy.Time.now())
        header.frame_id = self.camera_frame
        image_temp.header = header
        self.image_pub.publish(image_temp)


def main():
    rospy.init_node('yolov5_ros', anonymous=True)
    yolo_dect = Yolo_Dect()
    #rospy.spin()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rate.sleep()


if __name__ == "__main__":

    main()
