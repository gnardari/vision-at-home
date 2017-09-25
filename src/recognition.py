import os

from timeit import default_timer as timer
from object_detection.utils import label_map_util
import numpy as np
import tensorflow as tf

class ObjectDetector(object):
    def __init__(self, graph_path, label_map_path, num_classes):
        # Graph initialization
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            # Get graph variables
            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

        # Create session
        self.sess = tf.Session(graph=self.graph)

        # Label Map Setup
        label_map = label_map_util.load_labelmap(label_map_path)
        categories = label_map_util.convert_label_map_to_categories(
                                    label_map, max_num_classes=num_classes)
        self.category_idx = label_map_util.create_category_index(categories)


    def detect(self, image):
        image_expanded = np.expand_dims(image, axis=0)
        with self.graph.as_default():
            (boxes, scores, classes, num_detections) = self.sess.run(
                    [self.boxes, self.scores, self.classes, self.num_detections],
                   feed_dict={self.image_tensor: image_expanded})

        top_score_id = np.argmax(scores[0])
        top_class_id = classes[0][top_score_id]
        category = self.category_idx[top_class_id]['name']
        score = scores[0][top_score_id]
        box = boxes[0][top_score_id]

        return box, score, category
