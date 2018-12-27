from concurrent import futures
import io
import logging
from os import path

from ml_serving.drivers import driver
from ml_serving.utils import helpers
import numpy as np
from PIL import Image
import tensorflow as tf

import config
import dataset
import model as caption_model
import label_map_util


LOG = logging.getLogger(__name__)
PARAMS = {
    'model-file': '289999.npy',
    'vocabulary-file': 'vocabulary.csv',
    'vocabulary-size': 5000,
    'max_boxes': 50,
    'threshold': 0.2,
    'label_map': '',
    'face_model': '',
    'face_threshold': 0.3,
    'pose_serving_addr': '',  # host:port
}
session = None
caption_generator = None
category_index = None
face_serving = None


def init_hook(**params):
    global PARAMS
    PARAMS.update(params)

    caption_init(**params)
    detection_init(**params)
    face_init(**params)


def caption_init(**params):
    model_file = PARAMS['model-file']
    vocabulary_file = PARAMS['vocabulary-file']
    if not path.exists(model_file) and not path.exists(vocabulary_file):
        return

    LOG.info('------------------------')
    LOG.info('Loading image-caption model...')

    PARAMS['vocabulary-size'] = int(PARAMS['vocabulary-size'])

    global session
    global caption_generator
    global vocabulary

    LOG.info('[Captions] Loading vocabulary at %s...' % vocabulary_file)

    vocabulary = dataset.Vocabulary(
        PARAMS['vocabulary-size'],
        vocabulary_file,
    )

    cfg = config.Config()
    cfg.phase = 'test'
    cfg.beam_size = 3
    sess = tf.Session()
    caption_generator = caption_model.CaptionGenerator(cfg)

    LOG.info('[Captions] Loading model at %s...' % model_file)
    caption_generator.load(sess, model_file)
    tf.get_default_graph().finalize()
    session = sess

    LOG.info('Loaded.')
    LOG.info('------------------------')


def detection_init(**params):
    threshold = params.get('threshold')
    max_boxes = params.get('max_boxes')
    if threshold:
        PARAMS['threshold'] = float(threshold)

    if max_boxes:
        PARAMS['max_boxes'] = int(max_boxes)

    label_map_path = params.get('label_map')
    if not label_map_path:
        raise RuntimeError(
            'Label map required. Provide path to label_map via'
            ' -o label_map=<label_map.pbtxt>'
        )

    LOG.info('Loading label map from %s...' % label_map_path)
    label_map = label_map_util.load_labelmap(label_map_path)
    max_num_classes = max([item.id for item in label_map.item])
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes
    )
    global category_index
    category_index = label_map_util.create_category_index(categories)
    LOG.info('Loaded.')


def face_init(**params):
    threshold = params.get('face_threshold')
    if threshold:
        PARAMS['face_threshold'] = float(threshold)
    # Load driver
    face_model = params.get('face_model')
    if not face_model:
        return

    LOG.info('------------------------')
    LOG.info('Loading face model at %s...' % face_model)
    drv = driver.load_driver('openvino')
    # Instantiate driver
    global face_serving
    face_serving = drv()
    face_serving.load_model(
        face_model,
        device='CPU',
        flexible_batch_size=True,
    )

    LOG.info('Loaded.')
    LOG.info('------------------------')


def load_image(image):
    """ Preprocess an image. """
    # if self.bgr:
    #     temp = image.swapaxes(0, 2)
    #     temp = temp[::-1]
    #     image = temp.swapaxes(0, 2)
    scale_shape = np.array([224, 224], np.int32)
    crop_shape = np.array([224, 224], np.int32)
    mean = np.array([104.00698793, 116.66876762, 122.67891434])

    image = image.resize((scale_shape[0], scale_shape[1]), Image.LANCZOS)
    image = np.array(image)
    offset = (scale_shape - crop_shape) / 2
    offset = offset.astype(np.int32)
    image = image[offset[0]:offset[0]+crop_shape[0],
                  offset[1]:offset[1]+crop_shape[1]]
    image = image - mean
    return image


def preprocess(inputs, ctx, **kwargs):
    image = inputs.get('input')
    if image is None:
        raise RuntimeError('Missing "input" key in inputs. Provide an image in "input" key')

    ctx.raw_image = image[0]
    image = Image.open(io.BytesIO(image[0]))
    image = image.convert('RGB')
    ctx.image = image
    preprocessed = load_image(image)
    ctx.caption_image = np.array(preprocessed, np.float32)
    ctx.np_image = np.array(image)

    data = image.resize((300, 300), Image.ANTIALIAS)
    data = np.array(data).transpose([2, 0, 1]).reshape(1, 3, 300, 300)
    # convert to BGR
    data = data[:, ::-1, :, :]
    ctx.face_image = data

    input_key = list(kwargs.get('model_inputs').keys())[0]
    return {input_key: [ctx.np_image]}


def get_caption_output(ctx):
    if caption_generator is None:
        return {}

    captions = []
    # scores = []

    caption_data = caption_generator.beam_search_images(
        session, [ctx.caption_image], vocabulary
    )

    word_idxs = caption_data[0][0].sentence
    # score = caption_data[0][0].score
    caption = vocabulary.get_sentence(word_idxs)
    captions.append(caption)
    # scores.append(score)

    return {
        'captions': captions,
        # 'caption_scores': scores,
    }


def get_detection_output(outputs, index=None):
    detection_boxes = outputs["detection_boxes"].reshape([-1, 4])
    detection_scores = outputs["detection_scores"].reshape([-1])
    detection_classes = np.int32((outputs["detection_classes"])).reshape([-1])

    max_boxes = PARAMS['max_boxes']
    threshold = PARAMS['threshold']

    detection_scores = detection_scores[np.where(detection_scores > threshold)]
    if len(detection_scores) < max_boxes:
        max_boxes = len(detection_scores)
    else:
        detection_scores = detection_scores[:max_boxes]

    detection_boxes = detection_boxes[:max_boxes]
    detection_classes = detection_classes[:max_boxes]

    classes = [index[i]['name'] for i in detection_classes]

    return {
        'detection_boxes': detection_boxes,
        'detection_classes': classes,
        'detection_scores': detection_scores,
    }


def get_face_output(outputs, ctx):
    if face_serving is None:
        return {}

    input_name = list(face_serving.inputs.keys())[0]

    image = ctx.face_image
    feed_dict = {input_name: image}
    face_out = face_serving.predict(feed_dict)
    face_out = list(face_out.values())[0].reshape([-1, 7])

    # Select boxes where confidence > factor
    bboxes_raw = face_out[face_out[:, 2] > PARAMS['face_threshold']]
    if bboxes_raw is None:
        return {}

    scores = bboxes_raw[:, 2]
    boxes = bboxes_raw[:, 3:7]

    return {
        'face_boxes': boxes,
        'face_scores': scores,
    }


def get_pose_output(ctx):
    pose_addr = PARAMS['pose_serving_addr']
    result = {}
    if pose_addr:
        pose_out = helpers.predict_grpc({'inputs': [ctx.raw_image]}, pose_addr)
        result['pose_boxes'] = pose_out['detection_boxes']
        result['pose_scores'] = pose_out['detection_scores']
        result['pose_classes'] = pose_out['detection_classes']
    return result


def postprocess(outputs, ctx):
    result = {}
    tpool = futures.ThreadPoolExecutor(max_workers=4)

    def process():
        caption_out = get_caption_output(ctx)
        detection_out = get_detection_output(outputs, index=category_index)
        face_out = get_face_output(outputs, ctx)
        result.update(caption_out)
        result.update(detection_out)
        result.update(face_out)

    def poses():
        pose_out = get_pose_output(ctx)
        result.update(pose_out)

    tpool.submit(process)
    tpool.submit(poses)
    tpool.shutdown()

    return result
