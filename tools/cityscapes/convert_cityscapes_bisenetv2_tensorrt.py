#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/12/13
# @Author  : Janghoo Lee, ProtossDragoon
# @Site    : https://github.com/ProtossDragoon/bisenetv2-tensorflow
# @File    : convert_cityscapes_bisenetv2_tensorrt.py
# @IDE: Google COLAB, VSCode
"""
TF-TensorRT Optimization bisenetv2 on cityspaces dataset
"""
# import essential library
import argparse
import tensorflow as tf
assert tf.__version__.startswith('1')
print(tf.__version__)
import cv2
import numpy as np
from tensorflow.python.compiler.tensorrt import trt_convert as trt


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--demo_image_path', type=str, help='The input image for demo',
                        default='./data/test_image/KUscapes/10.59.46.png')
    parser.add_argument('-f', '--frozen_graph_path', type=str, help='The model frozen graph file path',
                        default='./checkpoint/bisenetv2_cityscapes_frozen.pb')
    parser.add_argument('-c', '--is_env_colab', action="store_true", help='Set if your env is Google COLAB')

    return parser.parse_args()
    

def preprocess_image(src_image, input_tensor_size, print_shape=False):
    """bisenet 의 input 에 공통적으로 적용되는 전처리입니다.

    Args:
        src_image (nparray):
        input_tensor_size (tuple): 

    Returns:
        nparray: 정규화되고, 채널이 RGB 또는 GRB 로 변환된 image numpy array
    """
    output_image = src_image[:, :, (2, 1, 0)]
    output_image = cv2.resize(
        output_image,
        dsize=(input_tensor_size[0], input_tensor_size[1]),
        interpolation=cv2.INTER_LINEAR
    )
    output_image = output_image.astype('float32') / 255.0
    img_mean = np.array([0.5, 0.5, 0.5]).reshape((1, 1, len([0.5, 0.5, 0.5])))
    img_std = np.array([0.5, 0.5, 0.5]).reshape((1, 1, len([0.5, 0.5, 0.5])))
    output_image -= img_mean
    output_image /= img_std
    if print_shape:
        print('shape :', output_image.shape)

    return output_image


def print_default_graph_nodes_name():
    with tf.get_default_graph().as_default() as graph:
        # graph node 의 이름을 조회하기 위해서는 graphdef 자료형이 준비되어 있어야 함.
        for node in graph.as_graph_def().node:
            print(node.name)
        print('\n\ntotal {} nodes were detected'.format(len(graph.as_graph_def().node)))
            

def get_trt_graph_def(frozen_graph_path):
    """tensorRT Graphdef 를 생성합니다.
    
    Args:
        frozen_graph_def (filepath): frozen graph (.pb) 파일의 경로

    Returns:
        tensorRT Graphdef: TensorRT Graph
    """

    # default_graph 초기화
    tf.reset_default_graph()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

        # First deserialize your frozen graph:
        with tf.gfile.GFile(frozen_graph_path, 'rb') as f:
            frozen_graph = tf.GraphDef()
            frozen_graph.ParseFromString(f.read())

        # graphdef 를 default_graph 에 추가하기
        tf.import_graph_def(frozen_graph, name="frozen")

        # node list print
        print_default_graph_nodes_name()

        # Now you can create a TensorRT inference graph from your
        # frozen graph:
        # trt.TrtGraphConverter + converter.convert() == trt.create_inference_graph()
        converter = trt.TrtGraphConverter(
            input_saved_model_dir=None, # input 방법 1
            input_saved_model_tags=None, # input 방법 2
            input_saved_model_signature_key=None, # input 방법 3
            input_graph_def=frozen_graph, # input 방법 4
            nodes_blacklist=['final_output'],
            max_batch_size=1,
            max_workspace_size_bytes=1<<30,
            precision_mode="FP32",
            minimum_segment_size=3,
            is_dynamic_op=False,
            maximum_cached_engines=1,
            use_calibration=True
            )
        trt_graph_def = converter.convert()

    return trt_graph_def


def trt_inference_demo(trt_graph_def, demo_image_path, environment_colab=True):
    """tensorRT graph 를 가져와서, tensorRT graph 로 추론합니다.

    Args:
        trt_graph_def (tensorRT Graphdef): tensorrt 로 conversion (converter.convert()) 한 graphdef
        demo_image_path (path): 이미지 경로
        environment_colab (bool, optional): Google Colab 에서는 imshow 가 지원되지 않기 때문에, Colab 일 경우에만 False 합니다. Defaults to True.

    Returns:
        nparray: inference result 를 담고 있는 numpy array
    """
    
    # default_graph 초기화
    tf.reset_default_graph()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        src_imge = cv2.imread(demo_image_path, cv2.IMREAD_COLOR)
        preprocessed_image = preprocess_image(src_imge, [1024, 512])

        # input_map = {'input_tensor':input_tensor}
        outputs = ['final_output:0', 'BiseNetV2/prob:0']

        # Import the TensorRT graph into a new graph and run:
        output_nodes = tf.import_graph_def(
            trt_graph_def,
            # input_map=input_map,
            return_elements=outputs,
            name="tftrt"
        )
            
        # print('input tensor :', input_tensor)
        print('output node :', output_nodes)

        # node list print
        print_default_graph_nodes_name()

        # run inference
        result = sess.run(output_nodes, feed_dict={'tftrt/input_tensor:0':[preprocessed_image]})
        for i in range(len(result)):
            print('result {} - tensor name <{}> shape : {}'.format(i, outputs[i], result[i].shape))

    # visualize
    if environment_colab:
        import matplotlib.pyplot as plt
        # hard coded area
        plt.figure(figsize=[13,20])
        result[0] = np.array(result[0], dtype=np.float32)
        result[1] = np.array(result[1], dtype=np.float32)
        plt.subplot(4,1,1)
        plt.imshow(np.array(src_imge, dtype=np.float32))
        plt.title('original')
        plt.subplot(4,2,1)
        plt.imshow(result[0])
        plt.title('result [0]')
        plt.subplot(4,3,1)        
        plt.imshow(result[1][0,:,:,0:3])
        plt.title('result [1] [RGB] road/sidewalk/building')
        plt.subplot(4,4,1)
        plt.imshow(result[1][0,:,:,11:14])
        plt.title('result [1] [RGB] person/rider/car')
        # ---

    else:
        # hard coded area
        result[0] = np.array(result[0], dtype=np.float32)
        result[1] = np.array(result[1], dtype=np.float32)
        cv2.imshow('original', src_imge)
        cv2.imshow('result [0]', result[0])
        cv2.imshow('result [1] [RGB] road/sidewalk/building', cv2.cvtColor(result[1][0,:,:,0:3], cv2.COLOR_BGR2RGB))
        cv2.imshow('result [1] [RGB] person/rider/car', cv2.cvtColor(result[1][0,:,:,11:14], cv2.COLOR_BGR2RGB))
        cv2.waitKey(0) & 0xff == ord('q')
        # ---
    
    return result
        

if __name__ == '__main__':
    """
    test code
    """
    args = init_args()

    trt_graph_def = get_trt_graph_def(args.frozen_graph_path)
    result = trt_inference_demo(trt_graph_def, args.demo_image_path, environment_colab=args.is_env_colab)
