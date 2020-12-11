import tensorflow as tf
assert tf.__version__.startswith('1')
print(tf.__version__)
import cv2
import numpy as np

from tensorflow.python.compiler.tensorrt import trt_convert as trt

def preprocess_image(src_image, input_tensor_size):
    """

    :param src_image:
    :param input_tensor_size:
    :return:
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
    return output_image


def main(FROZEN_GRAPH_PATH='checkpoint/bisenetv2_cityscapes_frozen.pb'):
    # 이 부분은 pb 를 tensorRT 로 변환하는 부분이다.
    tf.reset_default_graph()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

        # First deserialize your frozen graph:
        with tf.gfile.GFile(FROZEN_GRAPH_PATH, 'rb') as f:
            frozen_graph = tf.GraphDef()
            frozen_graph.ParseFromString(f.read())

        tf.import_graph_def(frozen_graph, name="frozen") # graph 가져오기

        with tf.get_default_graph().as_default() as graph :
            print('---------------')
            for node in graph.as_graph_def().node :
                print(node.name)
            print('---------------')
            input_tensor = graph.get_tensor_by_name("frozen/input_tensor:0")
        print('input tensor :', input_tensor)
        print('---------------')

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
        trt_graph = converter.convert()
    

    # 이 부분은 tensorRT graph 를 가져와서, tensorRT graph 로 추론하는 부분이다.
    tf.reset_default_graph()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        image_path = 'data/test_image/KUscapes/10.59.46.png'
        src_imge = cv2.imread(image_path, cv2.IMREAD_COLOR)
        preprocessed_image = preprocess_image(src_imge, [1024, 512])


        # input_map = {'input_tensor':input_tensor}
        outputs = ['final_output:0', 'BiseNetV2/prob:0']

        # Import the TensorRT graph into a new graph and run:
        output_node = tf.import_graph_def(
            trt_graph,
            # input_map=input_map,
            return_elements=outputs,
            name="tftrt"
        )
            
        # print('input tensor :', input_tensor)
        print('output node :', output_node)

        with tf.get_default_graph().as_default() as graph :
            print('---------------')
            for node in graph.as_graph_def().node :
                print(node.name)
            print('---------------')


        '''
        input_tensor = tf.placeholder(
            dtype=tf.float32,
            shape=[1, input_tensor_size[1], input_tensor_size[0], 3],
            name='input_tensor'
        )
        '''

        print('--------------- image preprocessing')
        print('shape :', preprocessed_image.shape)
        print('---------------')

        result = sess.run(output_node,
                        feed_dict={'tftrt/input_tensor:0':[preprocessed_image]}
                        )
        print(result)

    for i in range(len(result)):
        print('result {} - tensor name <{}> shape : {}'.format(i, outputs[i], result[i].shape))

    # hard coded area
    result[0] = np.array(result[0], dtype=np.float32)
    result[1] = np.array(result[1], dtype=np.float32)
    cap_li = []
    cap_li.append(cv2.imshow('original', cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)))
    cap_li.append(cv2.imshow('result [0]', result[0]))
    CH = 0
    cap_li.append(cv2.imshow('result [1] [RGB] road/sidewalk/building', cv2.cvtColor(result[1][0,:,:,0:3], cv2.COLOR_BGR2RGB)))
    cap_li.append(cv2.imshow('result [1] [RGB] person/rider/car', cv2.cvtColor(result[1][0,:,:,11:14], cv2.COLOR_BGR2RGB)))

    if cv2.waitKey(0) & 0xff == ord('q'):
        for cap in cap_li:
            cap.release()

if __name__ == '__main__':
    """
    test code
    """
    main()