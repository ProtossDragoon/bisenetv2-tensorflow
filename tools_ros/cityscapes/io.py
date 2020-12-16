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
import time

# import ROS package
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rospy

# import bisenetv2-tensorflow module
from tools.cityscapes.convert_cityscapes_bisenetv2_tensorrt import preprocess_image, print_default_graph_nodes_name
from tools.cityscapes.convert_cityscapes_bisenetv2_tensorrt import get_trt_graph_def


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--frozen_graph_path', type=str, help='The model frozen graph file path',
                        default='checkpoint/bisenetv2_cityscapes_frozen.pb')

    return parser.parse_args()


def time_logger(time_dict):
    total_time = sum(list(time_dict.values()))
    for idx, key in enumerate(list(time_dict.keys())):
        print('[time] {}.{} : ({}%) {}s'.format(idx, key, round(time_dict[key]*100 / total_time, 3), round(time_dict[key], 3)))
    print('[time] {}.total : {}s ({} FPS)'.format(idx + 1, round(total_time, 3), round(1/total_time, 2)))


def datatype_converter_for_publisher(modeloutput, dtype_result, dtype_target):
    print('result i : {}({}) -> '.format(modeloutput.shape, modeloutput.dtype), end='')
    if str(dtype_result) == 'float32' and str(dtype_target) == 'uint8':
        modeloutput = modeloutput.astype(np.float64) / modeloutput.max()
        modeloutput = (255 * modeloutput).astype(np.uint8)
        print('publish : {}({})'.format(modeloutput.shape, modeloutput.dtype))
        return modeloutput
    elif str(dtype_result) == 'float16':
        raise NotImplementedError
    elif str(dtype_result) == 'uint8':
        raise NotImplementedError        
    else :
        raise NotImplementedError


def callback(data):
    global output_names
    global output_nodes
    global pub
    global bridge
    global runtime_session
    global tracking_output_node_list

    cv2_img = bridge.imgmsg_to_cv2(data, desired_encoding = 'passthrough')
    print('recieved image topic : {}'.format(cv2_img.dtype))

    # TODO : time checker refactoring with decorator
    time_dict = {}

    t_start_preprocess = time.time()
    preprocessed_image = preprocess_image(cv2_img, [1024, 512]) # FIXME : hardcoded size
    t_end_preprocess = time.time()
    time_dict['preprocessing'] = t_end_preprocess - t_start_preprocess

    t_start_inference = time.time()
    result = runtime_session.run(output_nodes, feed_dict={'tftrt/input_tensor:0':[preprocessed_image]})
    t_end_inference = time.time()
    time_dict['inference'] = t_end_inference - t_start_inference

    # result   : list
    # result[i]: np.ndarray
    for i in range(len(result)):
        print('result {} - tensor name <{}> shape : {}({})'.format(i, tracking_output_node_list[i], result[i].shape, result[i].dtype))
    
    ch = 0
    interested_classes_segmentation_result = result[1][0,:,:,ch:ch+3]

    t_start_dtypeconvert = time.time()
    res = datatype_converter_for_publisher(interested_classes_segmentation_result, result[1].dtype, 'uint8')    
    t_end_dtypeconvert = time.time()
    time_dict['dtypeconvert'] = t_end_dtypeconvert - t_start_dtypeconvert

    t_start_bridge = time.time()
    ku_img_msg = bridge.cv2_to_imgmsg(res, encoding = 'passthrough')
    t_end_bridge = time.time()
    time_dict['bridge'] = t_end_bridge - t_start_bridge

    print('pubslished image topic : {}({})'.format(res.shape, res.dtype))
    
    pub.publish(ku_img_msg)

    time_logger(time_dict)



def run_segmentation(trt_graph_def):
    
    # Import the TensorRT graph into a new graph and run:
    global output_names
    global output_nodes
    global pub
    global bridge
    global runtime_session
    global tracking_output_node_list
    bridge = CvBridge()

    tf.reset_default_graph()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        output_names = tracking_output_node_list
        output_nodes = tf.import_graph_def(
            trt_graph_def,
            # input_map=input_map,
            return_elements=tracking_output_node_list,
            name="tftrt"
        )

    rospy.init_node('segmentation', anonymous=True)
        # NOTE: the name must be a base name, i.e. it cannot contain any slashes "/".
        # anonymous = True ensures that your node has a unique name by adding random numbers to the end of NAME.
    pub = rospy.Publisher('segmentation_chatter_pub', Image, queue_size=10)
    runtime_session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    rospy.Subscriber('/camera/color/image_raw', Image, callback)
    rospy.spin()
    runtime_session.close()


def main(args):
    global runtime_session
    global tracking_output_node_list 

    try:
        trt_graph_def = get_trt_graph_def(args.frozen_graph_path)
        print_default_graph_nodes_name()
        tracking_output_node_list = ['final_output:0', 'BiseNetV2/prob:0']        
        run_segmentation(trt_graph_def)

    except rospy.ROSInterruptException:
        if not runtime_session._closed:
            runtime_session.close()
            print('Session closed successfully.')
        # this catches a rospy.ROSInterruptException exception
        # 해석하면.. 정상적 종료에 대해서 오류메시지 출력하고 그러지 않겠다는 뜻.
        pass


if __name__ == '__main__':
    args = init_args()
    main(args)