#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/12/15
# @Author  : Janghoo Lee, ProtossDragoon
# @Site    : https://github.com/ProtossDragoon/bisenetv2-tensorflow
# @File    : convert_cityscapes_bisenetv2_tensorrt.py
# @IDE: Google COLAB, VSCode
# @Env: NVIDIA Jetson, ROS Melodic, Ubuntu 18.04

"""
Set Environment for Google COLAB, VSCode, ROS
"""
import os
p = os.path.dirname(os.path.abspath(__file__))
print('Current File Path : {}'.format(p))
hard_coded_project_root_path = os.path.abspath(os.path.join(p, os.pardir, os.pardir))
print('Project Root Path : {} (Hardcoded)'.format(hard_coded_project_root_path))
import sys
if sys.path[0] != hard_coded_project_root_path:
    sys.path.insert(0, hard_coded_project_root_path)

"""
Test bisenetv2 on cityspaces dataset
"""
# import essential library
import argparse
import tensorflow as tf
assert tf.__version__.startswith('1')
print(tf.__version__)
import cv2
import numpy as np

# import ROS package
from cv_bridge import CvBridge
bridge = CvBridge()
from sensor_msgs.msg import Image
import rospy

# import bisenetv2-tensorflow module
from tools.cityspaces.convert_cityscapes_bisenetv2_tensorrt import preprocess_image, print_default_graph_nodes_name
from tools.cityspaces.convert_cityscapes_bisenetv2_tensorrt import get_trt_graph_def


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--frozen_graph_path', type=str, help='The model frozen graph file path',
                        default='checkpoint/bisenetv2_cityscapes_frozen.pb')

    return parser.parse_args()


def callback(data):
    global output_names
    global output_nodes
    cv2_img = bridge.imgmsg_to_cv2(data, desired_encoding = 'passthrough')
    preprocessed_image = preprocess_image(cv2_img)
    result = sess.run(output_nodes, feed_dict={'tftrt/input_tensor:0':[preprocessed_image]})
    for i in range(len(result)):
        print('result {} - tensor name <{}> shape : {}'.format(i, outputs[i], result[i].shape))
    

    # ---------- 고쳐야함
    pub = rospy.Publisher('image_chatter_pub', Image, queue_size=10)
    rospy.init_node('segmentation_node', anonymous=True)
        # NOTE: the name must be a base name, i.e. it cannot contain any slashes "/".
        # anonymous = True ensures that your node has a unique name by adding random numbers to the end of NAME.
    rate = rospy.Rate(10) # 10hz
    bridge = CvBridge()

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        print(type(frame))
        cv2.imshow('publisher', frame)
        ku_img_msg = bridge.cv2_to_imgmsg(frame, encoding = 'passthrough')
        # rospy.loginfo(ku_img_msg) # which performs triple-duty: the messages get printed to screen, it gets written to the Node's log file, and it gets written to rosout. rosout is a handy tool for debugging
        pub.publish(ku_img_msg)
        rate.sleep()    
    # ------------


def run_segmentation(trt_graph_def, tracking_output_node_list=['final_output:0']):
    tf.reset_default_graph()
    
    # Import the TensorRT graph into a new graph and run:
    global output_names
    global output_nodes
    output_names = tracking_output_node_list
    output_nodes = tf.import_graph_def(
        trt_graph_def,
        # input_map=input_map,
        return_elements=tracking_output_node_list,
        name="tftrt"
    )

    rospy.init_node('image_listener_node', anonymous=True)
    rospy.Subscriber('/camera/color/image_raw', Image, callback)
    rospy.spin()


    

if __name__ == '__main__':
    args = init_args()
    try:
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            trt_graph_def = get_trt_graph_def(args.frozen_graph_path)
            print_default_graph_nodes_name()
            tracking_output_node_list = ['final_output:0', 'BiseNetV2/prob:0']
            run_segmentation(trt_graph_def, tracking_output_node_list=tracking_output_node_list)

    except rospy.ROSInterruptException:
        # this catches a rospy.ROSInterruptException exception
        # 해석하면.. 정상적 종료에 대해서 오류메시지 출력하고 그러지 않겠다는 뜻.
        pass
