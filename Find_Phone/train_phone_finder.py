import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Grab path to current working directory
CWD_PATH = os.getcwd()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_graph(image):
    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'find_mobile_graph'




    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,'training','object-detection.pbtxt')

    # Number of classes the object detector can identify
    NUM_CLASSES = 1

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)

    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    image_expanded = np.expand_dims(image, axis=0)

    boxes = sess.run(
        detection_boxes,
        feed_dict={image_tensor: image_expanded})

    return boxes

"""def main():

    IMAGE_NAME = 'image13.jpg'
    # Path to image
    PATH_TO_IMAGE = os.path.join(CWD_PATH,'test_images',IMAGE_NAME)

    image = cv2.imread(PATH_TO_IMAGE)

    box = get_graph(image)
    # Perform the actual detection by running the model with the image as input
    print(box[0][0])

main()

"""
