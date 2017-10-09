import os
import json
import time
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from recognition import ObjectDetector
import numpy as np
import tensorflow as tf
import rospy
import cv2

GRAPH = os.path.join('graphs', 'rcnn_resnet101_frozen_inference_graph.pb')
LABELS = os.path.join('graphs', 'mscoco_label_map.pbtxt')
NCLASSES = 100

class RosTensorflow():
    def __init__(self, gpath, lmpath, nclasses, sthresh=0.8):
        self.detector = ObjectDetector(
                graph_path=gpath,
                label_map_path=lmpath,
                num_classes=nclasses)

        self.score_threshold = sthresh
        self._cv_bridge = CvBridge()
        self._sub = rospy.Subscriber('usb_cam/image_raw', Image, self.callback, queue_size=1)
        self._pub = rospy.Publisher('detection_result', String, queue_size=1)


    def callback(self, image_msg):
        cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "rgb8")

        s = time.time()
        result = self.detector.detect(cv_image)
        boxes, classes, scores = result
        tdiff = time.time() - s
        rospy.loginfo('Inference took: {:.3f} seconds'.format(tdiff))

        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes)
        scores = np.squeeze(scores)
        if scores[0] > self.score_threshold:
            self.detector.draw_detection(cv_image,
                                         boxes,
                                         classes.astype(np.int32),
                                         scores,
                                         'detection.jpg',
                                         score_thresh=self.score_threshold)

            top_ids = np.where(scores > self.score_threshold)
            classes = self.detector.ids_to_labels(classes[top_ids].tolist())
            scores = scores[top_ids].tolist()
            boxes = boxes[top_ids].tolist()

            msg = json.dumps({'classes': classes,
                              'scores': scores,
                              'boxes': boxes})

            rospy.loginfo(zip(classes, scores))
            # rospy.loginfo(msg)
            self._pub.publish(msg)
        else:
            rospy.loginfo('Weak detections: {}'.format(zip(classes[:3], scores[:3])))
            rospy.loginfo('Rotate robot')

    def main(self):
        rospy.loginfo('Starting Node...')
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('rostensorflow')
    rtf = RosTensorflow(GRAPH, LABELS, NCLASSES)
    rtf.main()
