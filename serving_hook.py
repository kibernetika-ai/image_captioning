import io
import logging
from os import path

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import tensorflow as tf

import config
import dataset
import model as caption_model
from im2txt import inference_wrapper
from im2txt import configuration
from im2txt.inference_utils import vocabulary as im2txt_vocab
from im2txt.inference_utils import caption_generator as im2txt_generator


LOG = logging.getLogger(__name__)
PARAMS = {
    'model-file': '289999.npy',
    'vocabulary-file': 'vocabulary.csv',
    'vocabulary-size': 5000,
}
IMAGE_CAPTIONING = 'image_captioning'
IM_2TXT = 'im2txt'
session = None
caption_generator = None
vocabulary = None
caption_type = IMAGE_CAPTIONING  # or 'im2txt'


def init_hook(**params):
    global PARAMS
    PARAMS.update(params)

    model_file = params['model-file']
    vocabulary_file = params['vocabulary-file']
    if not path.exists(model_file) and not path.exists(vocabulary_file):
        return

    LOG.info('------------------------')
    LOG.info('Loading image-caption model...')

    PARAMS['vocabulary-size'] = int(PARAMS['vocabulary-size'])

    global caption_type

    if model_file.endswith(".npy"):
        caption_type = IMAGE_CAPTIONING
        load_image_captioning(model_file, vocabulary_file)
    else:
        caption_type = IM_2TXT
        load_im2txt(model_file, vocabulary_file)

    LOG.info('Loaded.')
    LOG.info('------------------------')

    try:
        ImageFont.truetype('Roboto-Bold.ttf', 42)
        print('Loaded Roboto-Bold.ttf.')
    except:
        ImageFont.load_default()
        print('Loaded default PIL font.')


def load_image_captioning(model_file, vocabulary_file):
    LOG.info('[Captions] Loading vocabulary at %s...' % vocabulary_file)

    global vocabulary
    global caption_generator
    global session
    vocabulary = dataset.Vocabulary(
        None,
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


def load_im2txt(model_file, vocabulary_file):
    # Build the inference graph.
    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(
            configuration.ModelConfig(),
            model_file,
        )
    g.finalize()

    global caption_generator
    global session
    global vocabulary

    # Create the vocabulary.
    vocabulary = im2txt_vocab.Vocabulary(vocabulary_file)

    # Create session
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    sess = tf.Session(graph=g, config=config_proto)
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    caption_generator = im2txt_generator.CaptionGenerator(model, vocabulary, beam_size=2)
    session = sess


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


def prepare_caption_image(image):
    # Resize
    caption_image = cv2.resize(image, (346, 346), interpolation=cv2.INTER_LANCZOS4)

    # Crop
    caption_image = caption_image[23:322, 23:322]

    # normalize
    caption_image = caption_image.astype(np.float32) / 255.0
    caption_image = (caption_image - 0.5) * 2.0

    return caption_image.reshape(1, *caption_image.shape)


def preprocess(inputs, ctx):
    image = inputs.get('input')
    if image is None:
        raise RuntimeError('Missing "input" key in inputs. Provide an image in "input" key')

    # ctx.caption_image = image[0]
    image = Image.open(io.BytesIO(image[0]))
    image = image.convert('RGB')
    ctx.image = image

    if caption_type == IMAGE_CAPTIONING:
        image = load_image(image)
        ctx.caption_image = np.array(image, np.float32)
    else:
        ctx.caption_image = prepare_caption_image(np.array(image))

    return {'fake': 'fake'}


def get_font(w, h, text):
    try:
        size = 300
        f = ImageFont.truetype('Roboto-Bold.ttf', size)
        size_found = False
        padding = w // 20
        expanding = h // 8

        while not size_found:
            x, y = f.getsize(text)
            if x <= w - padding * 2 and y <= expanding - h // 25:
                break
            size -= 1
            f = f.font_variant(size=size)
        print('Set font size to %s.' % size)
        return f
    except:
        return ImageFont.load_default()


def get_caption_output(ctx, array=False):
    if caption_generator is None:
        return {}

    captions = []
    if caption_type == IMAGE_CAPTIONING:
        caption_data = caption_generator.beam_search_images(
            session, [ctx.caption_image], vocabulary
        )

        word_idxs = caption_data[0][0].sentence
        caption = vocabulary.get_sentence(word_idxs)
        captions.append(caption)
    elif caption_type == IM_2TXT:
        raw_captions = caption_generator.beam_search(session, ctx.caption_image, array=array)
        for i, caption in enumerate(raw_captions):
            # Ignore begin and end words.
            sentence = [vocabulary.id_to_word(w) for w in caption.sentence[1:-1]]
            caption = " ".join(sentence)
            captions.append(caption)

    return {
        'captions': captions,
    }


def montage_caption(image, caption):
    use_pil = False
    if isinstance(image, Image.Image):
        use_pil = True
        w = image.size[0]
        h = image.size[1]
    else:
        w = image.shape[1]
        h = image.shape[0]
    expanding = h // 8

    if use_pil:
        montage = Image.new(mode='RGBA', size=(w, h + expanding), color='white')

        montage.paste(Image.fromarray(image), (0, expanding))
        draw = ImageDraw.Draw(montage)
        font = get_font(w, h, caption)

        text_size = font.getsize(caption)
    else:
        up = np.ones([expanding, w, 3], dtype=np.uint8) * 255
        montage = np.vstack([up, image])
        font = cv2.FONT_HERSHEY_TRIPLEX

        size = 5.0
        size_found = False
        padding = w // 25
        expanding = h // 8

        font_scale = size
        font_thickness = 1
        while not size_found:
            size_t = cv2.getTextSize(caption, font, font_scale, font_thickness)
            x, y = size_t[0][0], size_t[0][1]
            if x <= w - padding * 2 and y <= expanding - h // 25:
                break
            size -= 0.02
            font_scale = size
        print('Set font scale to %.3f.' % size)

        size = cv2.getTextSize(caption, font, font_scale, font_thickness)
        text_size = size[0][0], size[0][1]

    text_x = (w - text_size[0]) // 2
    text_y = (expanding - text_size[1]) // 2
    text_xy = (text_x, text_y)

    if use_pil:
        draw.text(text_xy, caption, font=font, fill='black')
    else:
        text_xy = (text_xy[0], text_xy[1] + text_size[1])
        font_thickness = int(np.ceil(font_scale))
        cv2.putText(
            montage,
            caption,
            text_xy,
            font, font_scale,
            color=(0, 0, 0),
            thickness=font_thickness,
            lineType=cv2.LINE_AA,
        )

    return montage


def postprocess(outputs, ctx):
    result = get_caption_output(ctx, array=True)

    montage = montage_caption(ctx.image, result['captions'][0])

    image_bytes = io.BytesIO()
    montage.convert('RGB').save(image_bytes, format='JPEG', quality=80)

    return {
        'output': image_bytes.getvalue(),
        'captions': result['captions'],
    }
