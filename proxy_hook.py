import logging

from ml_serving.utils import helpers

import detection_hook


LOG = logging.getLogger(__name__)
PARAMS = {
    'max_boxes': 50,
    'threshold': 0.2,
    'face_threshold': 0.3,
    'pose_threshold': 0.5,
    'remote_serving_addr': '',  # host:port
    'output_type': 'boxes',  # Or 'image'
    'line_thickness': 4,
}


def init_hook(**params):
    global PARAMS
    PARAMS.update(params)

    remote_serving = PARAMS['remote_serving_addr']
    if not remote_serving:
        raise RuntimeError('provide remote_serving_addr via -o remote_serving_addr.')


def preprocess(inputs, ctx, **kwargs):
    image = inputs.get('input')
    if image is None:
        raise RuntimeError('Missing "input" key in inputs. Provide an image in "input" key')

    inputs['output_type'] = 'boxes'

    return helpers.predict_grpc(inputs, PARAMS['remote_serving_addr'])


def get_pose_output(outputs, ctx):
    result = {}
    pose_out = outputs
    result['pose_boxes'] = pose_out['detection_boxes']
    result['pose_scores'] = pose_out['detection_scores']
    result['pose_classes'] = pose_out['detection_classes']
    return result


def postprocess(outputs, ctx):
    result = outputs

    if PARAMS['output_type'] == 'image':
        return detection_hook.image_output(ctx, result, PARAMS)

    return result
