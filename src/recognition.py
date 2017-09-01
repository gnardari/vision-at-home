import os
import numpy as np
import tensorflow as tf

class ObjectDetector(object):
    def __init__(self, graph_path, label_map_path=None, num_classes=None):
        # Graph initialization
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.Gfile(graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Create session
        self.sess = tf.Session()

        # Label Map Setup
        # self.label_map = label_map_util.load_labelmap(label_map_path)
        # self.categories = label_map_util.convert_label_map_to_categories(
        #                             label_map, max_num_classes=num_classes)
        # self.category_idx = label_map_util.create_category_index(self.categories)

    def detect(image, max_objects=3):
        with self.graph.as_default():
            image_expanded = np.expand_dims(image, axis=0)
            image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            scores = self.graph.get_tensor_by_name('detection_scores:0')
            classes = self.graph.get_tensor_by_name('detection_classes:0')
            num_detections = self.graph.get_tensor_by_name('num_detections:0')

            (boxes, scores, classes, num_detections) = self.sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_expanded})

        print('Classes: {}'.format(classes))
        print('Scores: {}'.format(scores))
