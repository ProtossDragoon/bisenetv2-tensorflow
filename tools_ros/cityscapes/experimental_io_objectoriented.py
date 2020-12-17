#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/12/17
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
Run with bisenetv2 trained on cityspaces dataset
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


class SegmentationStreamer(object):
    def __init__(self, tracking_output_nodes_name_list, trt_graph_def):
        super().__init__()

        # for tensorflow
        self.tracking_output_nodes = None
        self.tracking_output_nodes_name_list = tracking_output_nodes_name_list

        self.trt_graph_def = trt_graph_def

        self.runtime_session = None

        # for ROS
        self.node_name = None
        self.publish_topic_name = None
        self.subscribe_topic_name = None
        self.publisher = None
        self.subscriber = None
        self.bridge = CvBridge()

        # utils
        self.__time_dict = {}

    def __time_logger(self, time_dict=None):
        if time_dict is None :
            time_dict = self.__time_dict

        total_time = sum(list(time_dict.values()))
        for idx, key in enumerate(list(time_dict.keys())):
            print('[time] {}.{} : ({}%) {}s'.format(idx, key, round(time_dict[key]*100 / total_time, 3), round(time_dict[key], 3)))
        print('[time] {}.total : {}s ({} FPS)'.format(idx + 1, round(total_time, 3), round(1/total_time, 2)))

    def __cal_time(self, func_to_profile):
        def wrapper(*args, **kwargs):
            s = time.time()
            res = func_to_profile(args, kwargs)
            e = time.time()
            self.__time_dict[func_to_profile.__name__] = e - s
            return res
        return wrapper

    @__cal_time
    def _preprocess_image(self, *args, **kwargs):
        pass
        # return preprocess_image(args, kwargs)

    @__cal_time
    def _do_inference(self, output_nodes, feed_dict={}):
        return self.runtime_session.run(output_nodes, feed_dict=feed_dict)

    @__cal_time
    def _convert_dtype(self, modeloutput, from_dtype, to_dtype):
        print('result i : {}({}) -> '.format(modeloutput.shape, modeloutput.dtype), end='')
        if str(from_dtype) == 'float32' and str(to_dtype) == 'uint8':
            modeloutput = modeloutput.astype(np.float64) / modeloutput.max()
            modeloutput = (255 * modeloutput).astype(np.uint8)
            print('publish : {}({})'.format(modeloutput.shape, modeloutput.dtype))
            return modeloutput
        elif str(from_dtype) == 'float16':
            raise NotImplementedError
        elif str(from_dtype) == 'uint8':
            raise NotImplementedError        
        else :
            raise NotImplementedError

    @__cal_time
    def _convert_to_rostype(self, res, encoding='passthrough'):
        return self.bridge.cv2_to_imgmsg(res, encoding=encoding)

    @__cal_time
    def _publish(self, data):
        self.publisher.publish(data)

    def __callback(self, data):
        cv2_img = self.bridge.imgmsg_to_cv2(data, desired_encoding = 'passthrough')
        print('recieved image topic : {}'.format(cv2_img.dtype))

        preprocessed_image = self._preprocess_image(cv2_img, [1024, 512]) # FIXME : hardcoded size
        result             = self._do_inference(self.tracking_output_nodes, feed_dict={'tftrt/input_tensor:0':[preprocessed_image]})

        # result   : list
        # result[i]: np.ndarray
        for i in range(len(result)):
            print('result {} - tensor name <{}> shape : {}({})'.format(i, self.tracking_output_nodes_name_list[i], result[i].shape, result[i].dtype))
        
        ch = 0
        interested_classes_segmentation_result = result[1][0,:,:,ch:ch+3]
        interested_classes_segmentation_result = self._convert_dtype(interested_classes_segmentation_result, result[1].dtype, 'uint8')
        ku_img_msg  = self._convert_to_rostype(result, encoding = 'passthrough')
    
        print('pubslished image topic : {}({})'.format(interested_classes_segmentation_result.shape, interested_classes_segmentation_result.dtype))
        self._publish(ku_img_msg)
        self.__time_logger()

    def run_segmentation_node(self, node_name='segmentation', publish_topic_name='segmentation_chatter_pub', subscribe_topic_name='/camera/color/image_raw'):
        # Import the TensorRT graph into a new graph and run:
        self.node_name              = node_name
        self.publish_topic_name     = publish_topic_name
        self.subscribe_topic_name   = subscribe_topic_name

        tf.reset_default_graph()
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            self.output_nodes = tf.import_graph_def(
                self.trt_graph_def,
                # input_map=input_map,
                return_elements=self.tracking_output_nodes_name_list,
                name="tftrt"
            )

        rospy.init_node(self.node_name, anonymous=True)
            # NOTE: the name must be a base name, i.e. it cannot contain any slashes "/".
            # anonymous = True ensures that your node has a unique name by adding random numbers to the end of NAME.
        self.publisher       = rospy.Publisher(self.publish_topic_name, Image, queue_size=10)
        self.runtime_session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.subscriber      = rospy.Subscriber(self.subscribe_topic_name, Image, self.__callback)

        rospy.spin()
        self.runtime_session.close()


def main(args):
    try:
        trt_graph_def = get_trt_graph_def(args.frozen_graph_path)
        print_default_graph_nodes_name()
        tracking_output_nodes_name_list = ['final_output:0', 'BiseNetV2/prob:0']

        streamer = SegmentationStreamer(tracking_output_nodes_name_list, trt_graph_def)
        streamer.run_segmentation_node(node_name='segmentation', publish_topic_name='segmentation_chatter_pub', subscribe_topic_name='/camera/color/image_raw')

    except rospy.ROSInterruptException:
        if not streamer.runtime_session._closed:
            streamer.runtime_session.close()
            print('Session closed successfully.')
        # this catches a rospy.ROSInterruptException exception
        # 해석하면.. 정상적 종료에 대해서 오류메시지 출력하고 그러지 않겠다는 뜻.
        pass


if __name__ == '__main__':
    args = init_args()
    main(args)
