import base64
import io
import json
import logging

import cv2
import numpy as np
from PIL import Image

from xray import model


LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
PARAMS = {
    'data_dir': '',
}
index_word = {}
label_map = {}


def init_hook(**kwargs):
    global PARAMS
    PARAMS.update(kwargs)

    global index_word
    global label_map
    word_index = model.get_word_index(PARAMS)
    label_map = model.load_label_map(PARAMS)

    for k, v in word_index.items():
        index_word[v] = k


def preprocess(inputs, ctx):
    image = inputs.get('input')
    if image is None:
        raise RuntimeError('Missing "input" key in inputs. Provide an image in "input" key')

    if len(image.shape) == 0:
        image = [image.tolist()]

    image = cv2.imdecode(np.frombuffer(image[0], np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ctx.image = image
    ctx.resized = cv2.resize(image, (299, 299), interpolation=cv2.INTER_AREA)

    return {'images': np.reshape(ctx.resized, [1, 299, 299, 3])}


def postprocess(outputs, ctx):
    predictions = outputs['predictions']
    attentions = outputs['attention']
    LOG.info('attentions: {}'.format(attentions.shape))
    LOG.info('predictions: {}'.format(predictions.shape))

    captions = []
    # for i in predictions[0]:
    #     t = word_index.get(i, None)
    #     if t is None or t == '<end>':
    #         continue
    #
    #     caption = label_map.get(t)
    #     if caption is not None:
    #         captions.append(caption)

    img_base = Image.fromarray(ctx.resized)
    img_base.putalpha(255)
    img_base = img_base.convert('RGBA')
    table = []

    for i, pred in enumerate(predictions[0]):
        t = index_word.get(pred, None)
        if t is None or t == '<end>':
            continue

        caption = label_map.get(t)
        if caption is None:
            continue

        captions.append(caption)

        attention = np.resize(attentions[0][i][0], (8, 8)) * 255
        image = Image.fromarray(attention.astype(np.uint8))
        image.putalpha(int(255 * 0.6))
        image = image.resize((299, 299))
        comp = Image.alpha_composite(img_base, image.convert('RGBA'))
        image_bytes = io.BytesIO()
        comp.save(image_bytes, format='PNG')
        encoded = base64.encodebytes(image_bytes.getvalue()).decode()
        table.append(
            {
                'type': 'text',
                'name': caption,
                'prob': 1.,
                'image': encoded
            }
        )
    image_bytes = io.BytesIO()
    img_base.save(image_bytes, format='PNG')
    return {
        'output': image_bytes.getvalue(),
        #'caption_output': np.array(captions),
        'table_output': json.dumps(table),
    }
