import base64
import io
import json
import logging
import time

from ml_serving.drivers import driver
import numpy as np
from PIL import Image

import label_map_util
import serving_hook
import visualization_utils as vis_utils


LOG = logging.getLogger(__name__)
PARAMS = {
    'model-file': '289999.npy',
    'vocabulary-file': 'vocabulary.csv',
    'vocabulary-size': 5000,
    'max_boxes': 50,
    'threshold': 0.33,
    'label_map': '',
    'pose_label_map': '',
    'face_model': '',
    'face_threshold': 0.46,
    'pose_threshold': 0.49,
    'remote_serving_addr': '',  # host:port
    'output_type': 'boxes',  # Or 'image'
    'line_thickness': 4,
    # /opt/intel/computer_vision_sdk/deployment_tools/intel_models/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml
    'emotion_model': '',
    'draw_caption': False,
}
session = None
caption_generator = None
caption_type = 'image_captioning'  # or 'im2txt'
category_index = None
pose_index = None
face_serving = None
emotion_serving = None


def init_hook(**params):
    global PARAMS
    PARAMS.update(params)

    remote_serving = PARAMS['remote_serving_addr']
    if remote_serving:
        return

    caption_init(**PARAMS)
    detection_init(**PARAMS)
    face_init(**PARAMS)
    emotion_init(**PARAMS)
    pose_init(**PARAMS)


def caption_init(**params):
    serving_hook.init_hook(**params)

    if params.get('draw_caption') is not None:
        PARAMS['draw_caption'] = str(params['draw_caption']).lower() == 'true'


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


def pose_init(**params):
    threshold = params.get('pose_threshold')
    max_boxes = params.get('max_boxes')
    if threshold:
        PARAMS['pose_threshold'] = float(threshold)

    if max_boxes:
        PARAMS['max_boxes'] = int(max_boxes)

    label_map_path = params.get('pose_label_map')
    if not label_map_path:
        raise RuntimeError(
            'Label map required. Provide path to label_map via'
            ' -o pose_label_map=<label_map.pbtxt>'
        )

    LOG.info('Loading label map from %s...' % label_map_path)
    label_map = label_map_util.load_labelmap(label_map_path)
    max_num_classes = max([item.id for item in label_map.item])
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes
    )
    global pose_index
    pose_index = label_map_util.create_category_index(categories)
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


def emotion_init(**params):
    # Load driver
    emotion_model = params.get('emotion_model')
    if not emotion_model:
        return

    LOG.info('------------------------')
    LOG.info('Loading emotion model at %s...' % emotion_model)
    drv = driver.load_driver('openvino')
    # Instantiate driver
    global emotion_serving
    emotion_serving = drv()
    emotion_serving.load_model(
        emotion_model,
        device='CPU',
        flexible_batch_size=True,
    )

    LOG.info('Loaded.')
    LOG.info('------------------------')


def set_detection_params(inputs, ctx):
    detection_params = [
        'detect_faces',
        'detect_objects',
        'build_caption',
        'detect_poses',
    ]
    for param in detection_params:
        raw_value = inputs.get(param)
        if raw_value is not None:
            LOG.info('%s=%s', param, raw_value)
            value = raw_value[0]
        else:
            # True by default
            value = True

        setattr(ctx, param, value)


def rotate_by_exif(image, ctx=None):
    if "exif" not in image.info:
        return image

    orientation = image._getexif().get(274)
    if not orientation:
        return image

    rotated = True

    if orientation == 3:
        image = image.rotate(180, expand=True)
    elif orientation == 4:
        image = image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 5:
        image = image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 6:
        image = image.rotate(-90, expand=True)
    elif orientation == 7:
        image = image.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 8:
        image = image.rotate(90, expand=True)
    else:
        rotated = False

    if rotated and ctx is not None:
        # Set raw_image
        image_bytes = io.BytesIO()
        image.convert('RGB').save(image_bytes, format='JPEG', quality=80)
        ctx.raw_image = image_bytes.getvalue()

    return image


def preprocess_detection(inputs, ctx, **kwargs):
    t = time.time()

    image = inputs.get('input')
    if image is None:
        raise RuntimeError('Missing "input" key in inputs. Provide an image in "input" key')

    set_detection_params(inputs, ctx)
    output_type = inputs.get('output_type')
    if output_type is not None:
        if len(output_type.shape) < 1:
            ctx.output_type = output_type.tolist().decode()
        else:
            ctx.output_type = output_type[0].decode()
    else:
        ctx.output_type = PARAMS['output_type']

    ctx.raw_image = image[0]
    image = Image.open(io.BytesIO(image[0]))

    # Rotate if exif tags specified
    image = rotate_by_exif(image, ctx)

    image = image.convert('RGB')
    ctx.image = image
    if serving_hook.caption_type == serving_hook.IMAGE_CAPTIONING:
        preprocessed = serving_hook.load_image(image)
        ctx.caption_image = np.array(preprocessed, np.float32)

    ctx.np_image = np.array(image)

    data = image.resize((300, 300), Image.ANTIALIAS)
    ctx.pose_image = np.array(data)
    data = np.array(data).transpose([2, 0, 1]).reshape(1, 3, 300, 300)
    # convert to BGR
    data = data[:, ::-1, :, :]
    ctx.face_image = data

    input_key = list(kwargs.get('model_inputs').keys())[0]
    LOG.info('preprocess detection: %.3fms' % ((time.time() - t) * 1000))
    ctx.t = time.time()

    return {input_key: [ctx.np_image]}


def preprocess_poses(inputs, ctx):
    ctx.t = time.time()
    if ctx.detect_poses:
        # return {'inputs': [ctx.np_image]}
        return {'inputs': [ctx.pose_image]}
    else:
        return {'inputs': np.zeros([1, 10, 10, 3]), 'ml-serving-ignore': True}


def get_detection_output(outputs, index=None, threshold=PARAMS['threshold']):
    detection_boxes = outputs["detection_boxes"].reshape([-1, 4])
    detection_scores = outputs["detection_scores"].reshape([-1])
    detection_classes = np.int32((outputs["detection_classes"])).reshape([-1])

    max_boxes = PARAMS['max_boxes']

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
    boxes_copy = np.copy(boxes)
    boxes[:, 0], boxes[:, 1] = boxes_copy[:, 1], boxes_copy[:, 0]
    boxes[:, 2], boxes[:, 3] = boxes_copy[:, 3], boxes_copy[:, 2]

    return {
        'face_boxes': boxes,
        'face_scores': scores,
    }


emotions = [
    'neutral', 'happy', 'sad', 'surprise', 'anger'
]


def dict_zip(a, b):
    zipped = zip(a, b)
    return dict(zipped)


def get_emotion_output(image, face_boxes):
    if emotion_serving is None:
        return {}
    if len(face_boxes) == 0:
        return {}

    input_name = list(emotion_serving.inputs.keys())[0]

    input_images = np.zeros([len(face_boxes), 3, 64, 64])
    boxes = np.copy(face_boxes)
    boxes[:, 0] = boxes[:, 0] * image.height
    boxes[:, 2] = boxes[:, 2] * image.height
    boxes[:, 1] = boxes[:, 1] * image.width
    boxes[:, 3] = boxes[:, 3] * image.width
    for i, box in enumerate(boxes):
        ymin = box[0]
        xmin = box[1]
        ymax = box[2]
        xmax = box[3]
        # Crop
        resized = image.crop((xmin, ymin, xmax, ymax)).resize((64, 64))
        input_images[i] = np.array(resized).transpose([2, 0, 1])

    # Convert to BGR
    input_images = input_images[:, ::-1, :, :]
    feed_dict = {input_name: input_images}
    emotion_out = emotion_serving.predict(feed_dict)
    # Shape: [N, 5, 1, 1]
    emotion_prob = list(emotion_out.values())[0].reshape([-1, 5])

    # TODO: how to pass the value below from serving?
    # TODO: Tensor proto doesn't allow to pass structures like that.
    # TODO: It allows only one-typed many-dimensioned list.
    # [{'happy': 0.78, 'sad': 0.01, 'angry': 0.1}]
    # full_emotions_prob = list(map(dict_zip, [emotions] * len(emotion_prob), emotion_prob))

    emotions_max = np.array(emotions)[emotion_prob.argmax(axis=1)]

    return {
        'emotion_prob': emotion_prob,
        'emotion_max': emotions_max,
        'possible_emotions': emotions,
    }


def get_pose_output(outputs, ctx):
    result = {}
    pose_out = outputs
    result['pose_boxes'] = pose_out['detection_boxes']
    result['pose_scores'] = pose_out['detection_scores']
    result['pose_classes'] = pose_out['detection_classes']
    return result


def postprocess_detection(outputs, ctx):
    result = {}
    LOG.info('object-detection: %.3fms' % ((time.time() - ctx.t) * 1000))

    def process():
        if ctx.build_caption:
            t = time.time()
            caption_out = serving_hook.get_caption_output(ctx)
            result.update(caption_out)

            LOG.info('build caption: %.3fms' % ((time.time() - t) * 1000))

        if ctx.detect_objects:
            detection_out = get_detection_output(outputs, index=category_index)
            result.update(detection_out)

        if ctx.detect_faces:
            t = time.time()
            face_out = get_face_output(outputs, ctx)
            result.update(face_out)

            LOG.info('face-detection: %.3fms' % ((time.time() - t) * 1000))
            t = time.time()

            emotion_out = get_emotion_output(ctx.image, face_out['face_boxes'])
            result.update(emotion_out)

            LOG.info('emotion-detection: %.3fms' % ((time.time() - t) * 1000))

    process()
    ctx.result = result

    return result


def image_resize(image, width=None, height=None, inter=Image.ANTIALIAS):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.height, image.width

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        if h < height:
            return image
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        if w < width:
            return image
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = image.resize(dim, inter)

    # return the resized image
    return resized


def result_table_string(result_dict, ctx):
    table = []

    def crop_from_box(box, normalized_coordinates=True):
        left, right = box[1], box[3]
        top, bottom = box[0], box[2]
        if normalized_coordinates:
            left, right = left * ctx.image.width, right * ctx.image.width
            top, bottom = top * ctx.image.height, bottom * ctx.image.height

        cropped = ctx.image.crop((left, top, right, bottom))
        cropped = image_resize(cropped, width=256)
        image_bytes = io.BytesIO()
        cropped.convert('RGB').save(image_bytes, format='JPEG', quality=80)

        return image_bytes.getvalue()

    def append(type_, name, prob, image):
        encoded = image
        if image is not None:
            encoded = base64.encodebytes(image).decode()

        table.append(
            {
                'type': type_,
                'name': name,
                'prob': float(prob),
                'image': encoded
            }
        )

    if ctx.detect_objects:
        zipped = zip(
            result_dict['detection_classes'],
            result_dict['detection_scores'],
            result_dict['detection_boxes']
        )
        for name, prob, box in zipped:
            append('object', name, prob, crop_from_box(box))

    if ctx.detect_poses:
        zipped = zip(
            result_dict['pose_classes'],
            result_dict['pose_scores'],
            result_dict['pose_boxes']
        )
        for name, prob, box in zipped:
            append('pose', name, prob, crop_from_box(box))

    if ctx.build_caption:
        for caption in result_dict['captions']:
            append('caption', caption, 0.5, None)

    if ctx.detect_faces and len(result_dict.get('face_boxes', [])) > 0:
        for prob, box in zip(result_dict['face_scores'], result_dict['face_boxes']):
            append('face', 'face', prob, crop_from_box(box))

        emotion_max = result_dict['emotion_max']
        emotion_prob = result_dict['emotion_prob']
        for i, box in enumerate(result_dict['face_boxes']):
            append('emotion', emotion_max[i], max(emotion_prob[i]), crop_from_box(box))

    return json.dumps(table)


def postprocess_poses(outputs, ctx):
    result = ctx.result
    LOG.info('pose-detection: %.3fms' % ((time.time() - ctx.t) * 1000))

    if ctx.detect_poses:
        pose_out = get_detection_output(outputs, pose_index, PARAMS['pose_threshold'])
        pose_out = get_pose_output(pose_out, ctx)
        result.update(pose_out)

    if ctx.output_type == 'image':
        return image_output(ctx, result, PARAMS)

    result['table_output'] = result_table_string(result, ctx)
    result['caption_output'] = result.get('captions', [])
    return result


def image_output(ctx, result, params):
    t = time.time()

    table_string = result_table_string(result, ctx)

    if ctx.detect_poses:
        vis_utils.visualize_boxes_and_labels_on_image(
            ctx.image,
            result['pose_boxes'],
            result['pose_classes'],
            result['pose_scores'],
            None,
            use_normalized_coordinates=True,
            max_boxes_to_draw=params['max_boxes'],
            min_score_thresh=params['pose_threshold'],
            agnostic_mode=False,
            line_thickness=params['line_thickness'],
            skip_labels=False,
            skip_scores=False,
        )

    if ctx.detect_objects:
        vis_utils.visualize_boxes_and_labels_on_image(
            ctx.image,
            result['detection_boxes'],
            result['detection_classes'],
            result['detection_scores'],
            None,
            use_normalized_coordinates=True,
            max_boxes_to_draw=params['max_boxes'],
            min_score_thresh=params['threshold'],
            agnostic_mode=False,
            line_thickness=params['line_thickness'],
            skip_labels=False,
            skip_scores=False,
        )

    if ctx.detect_faces and len(result.get('face_boxes', [])) > 0:
        face_boxes = result['face_boxes']
        emotion_max = result['emotion_max']
        emotion_prob = result['emotion_prob']
        for i, box in enumerate(face_boxes):
            display_str = '%s: %d%%' % (emotion_max[i], int(max(emotion_prob[i]) * 100))
            vis_utils.draw_bounding_box_on_image(
                ctx.image,
                box[0],
                box[1],
                box[2],
                box[3],
                color=(250, 0, 250),
                thickness=params['line_thickness'],
                display_str_list=(display_str,),
                use_normalized_coordinates=True,
            )

    if ctx.build_caption and PARAMS['draw_caption']:
        if result['captions']:
            ctx.image = serving_hook.montage_caption(ctx.image, result['captions'][0])

    image_bytes = io.BytesIO()
    ctx.image.convert('RGB').save(image_bytes, format='JPEG', quality=80)

    LOG.info('render-image: %.3fms' % ((time.time() - t) * 1000))

    return {
        'output': image_bytes.getvalue(),
        'table_output': table_string,
        'caption_output': result.get('captions', [])
    }


preprocess = [
    preprocess_detection,
    preprocess_poses
]
postprocess = [
    postprocess_detection,
    postprocess_poses
]


def draw_rectangle(draw, coordinates, color, width=1):
    for i in range(width):
        rect_start = (coordinates[0][0] - i, coordinates[0][1] - i)
        rect_end = (coordinates[1][0] + i, coordinates[1][1] + i)
        draw.rectangle((rect_start, rect_end), outline=color)