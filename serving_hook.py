import io
import logging

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import tensorflow as tf

import config
import dataset
import model as caption_model

LOG = logging.getLogger(__name__)
PARAMS = {
    'model-file': '289999.npy',
    'vocabulary-file': 'vocabulary.csv',
    'vocabulary-size': 5000,
}
session = None
model = None
image_font = None


def init_hook(**params):
    global PARAMS
    PARAMS.update(params)

    PARAMS['vocabulary-size'] = int(PARAMS['vocabulary-size'])

    global session
    global model
    global vocabulary
    global image_font

    vocabulary = dataset.Vocabulary(
        PARAMS['vocabulary-size'],
        PARAMS['vocabulary-file'],
    )

    cfg = config.Config()
    cfg.phase = 'test'
    cfg.beam_size = 3
    sess = tf.Session()
    model = caption_model.CaptionGenerator(cfg)

    model.load(sess, PARAMS['model-file'])
    tf.get_default_graph().finalize()
    session = sess

    try:
        image_font = ImageFont.truetype('Roboto-Bold.ttf', 42)
        print('Loaded Roboto-Bold.ttf.')
    except:
        image_font = ImageFont.load_default()
        print('Loaded default PIL font.')


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

    image = Image.open(io.BytesIO(image[0]))
    image = image.convert('RGB')
    ctx.image = image
    image = load_image(image)
    ctx.np_image = np.array(image, np.float32)

    return {'fake': 'fake'}


def postprocess(outputs, ctx):
    captions = []
    scores = []

    caption_data = model.beam_search_images(session, [ctx.np_image], vocabulary)

    word_idxs = caption_data[0][0].sentence
    score = caption_data[0][0].score
    caption = vocabulary.get_sentence(word_idxs)
    captions.append(caption)
    scores.append(score)

    w = ctx.image.size[0]
    h = ctx.image.size[1]
    montage = Image.new(mode='RGBA', size=(w, h + 100), color='white')

    montage.paste(ctx.image, (0, 100))
    draw = ImageDraw.Draw(montage)

    text_width = image_font.getsize(caption)
    print(text_width)
    text_x = (w - text_width[0]) // 2
    text_xy = (text_x, 20)
    draw.text(text_xy, caption, font=image_font, fill='black')

    image_bytes = io.BytesIO()
    montage.save(image_bytes, format='PNG')

    return {
        'output': image_bytes.getvalue(),
        'captions': captions,
        'scores': scores,
    }
