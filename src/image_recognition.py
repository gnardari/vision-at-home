import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from recognition import ObjectDetector
import cv2
import numpy as np
import tensorflow as tf


class RosTensorflow():
    def __init__(self):
        self._cv_bridge = CvBridge()
        self._sub = rospy.Subscriber('camera', Image, self.callback, queue_size=1)
        self._pub = rospy.Publisher('detection_result', String, queue_size=1)

        self.detector = ObjectDetector(
                graph_path='',
                label_map_path='',
                num_classes='')

    def callback(self, image_msg):
        cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
        result = self.detector.detect(cv_image)
        boxes, scores, classes, num_detections = result

        # rospy.loginfo('%s (score = %.5f)' % (human_string, score))
        # self._pub.publish(human_string)

    def main(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('rostensorflow')
    rtf = RosTensorFlow()
    rtf.main()
