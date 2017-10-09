import os
import json
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from recognition import ObjectDetector
import numpy as np
import tensorflow as tf
import rospy
import cv2

GRAPH = os.path.join('graphs', 'fronzen_inference_graph.pb')
LABELS = os.path.join('graphs', 'mscoco_label_map.pbtxt')
NCLASSES = 100

class RosTensorflow():
    def __init__(self, gpath, lmpath, nclasses):
        self._cv_bridge = CvBridge()
        self._sub = rospy.Subscriber('camera', Image, self.callback, queue_size=1)
        self._pub = rospy.Publisher('detection_result', String, queue_size=1)

        self.detector = ObjectDetector(
                graph_path=gpath,
                label_map_path=lmpath,
                num_classes=nclasses)

    def callback(self, image_msg):
        cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "rgb8")
        result = self.detector.detect(cv_image)
        boxes, classes, scores = result

        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes)
        scores = np.squeeze(scores)
        if scores[0] > self.score_threshold:
            self.detector.draw_detection(boxes,
                                         classes.astype(np.int32),
                                         scores,
                                         'detection.jpg',
                                         use_normalized_coordinates=True,
                                         line_thickness=8)
            msg = json.dumps({'classes': classes, 'scores': scores, 'boxes': boxes})
            self._pub.publish(msg)
        else:
            rospy.loginfo('Weak detections: {}'.format(zip(classes[:3], scores[:3])))

    def main(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('rostensorflow')
    rtf = RosTensorFlow(GRAPH, LABELS, NCLASSES)
    rtf.main()
