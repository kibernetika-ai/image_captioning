import argparse
import json
import hashlib
import os
import re

import nltk
from nltk import tokenize
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import inception_v3


slim = tf.contrib.slim
nltk.download('punkt')
tf.logging.set_verbosity(tf.logging.INFO)
log = tf.logging


def _inception_v3_arg_scope(is_training=True,
                            weight_decay=0.00004,
                            stddev=0.1,
                            batch_norm_var_collection='moving_vars'):
    """Defines the default InceptionV3 arg scope.
    Args:
      is_training: Whether or not we're training the model.
      weight_decay: The weight decay to use for regularizing the model.
      stddev: The standard deviation of the trunctated normal weight initializer.
      batch_norm_var_collection: The name of the collection for the batch norm
        variables.
    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    batch_norm_params = {
        'is_training': is_training,
        # Decay for the moving averages.
        'decay': 0.9997,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # collection containing the moving mean and moving variance.
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection],
        }
    }
    normalizer_fn = slim.batch_norm

    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                activation_fn=tf.nn.relu6,
                normalizer_fn=normalizer_fn,
                normalizer_params=batch_norm_params) as sc:
            return sc


def _inception(inputs):
    with slim.arg_scope(_inception_v3_arg_scope(is_training=False)):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected, slim.batch_norm],
                trainable=True):
            with slim.arg_scope(
                    [slim.batch_norm, slim.dropout], is_training=False):
                netw, _ = inception_v3.inception_v3_base(
                    inputs,
                    scope='InceptionV3')
                return netw


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data', help='Dataset directory')
    parser.add_argument('--output', type=str, default='./features', help='Destination directory')
    parser.add_argument('--inception-path', type=str, default='./inception_v3.ckpt', help='Inception checkpoint')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data = pd.read_csv(args.data_dir + '/descriptions.csv')

    label_map = {}
    annotations = []
    descriptions = set()
    desc_count = 0
    for i, row in data.iterrows():
        preprocessed = row['description'].replace('. =', '.=').strip('["]').replace('. =', '.=').replace('", "',
                                                                                                         '.').strip()
        preprocessed = preprocessed.replace('Остеохондроз', ' Остеохондроз').replace('(', ' (')
        preprocessed = re.sub('([\w]{2})\.([\w]{2})', '\\1. \\2', preprocessed)
        row_descriptions = tokenize.sent_tokenize(preprocessed)
        row_descriptions = list(
            filter(lambda x: x and len(x) > 18, [s.strip().lower().strip('(?.') for s in row_descriptions])
        )

        hashes = []
        for desc in row_descriptions:
            dig = hashlib.sha256(desc.encode()).hexdigest()
            # Take only first N characters from hash to make it shorter
            dig = dig[:6]
            label_map[dig] = desc
            hashes.append(dig)

        image_id = row['image_name'][:row['image_name'].rfind('.')]
        annotations.append({'image_id': image_id, 'image_name': row['image_name'], 'caption': ' '.join(hashes)})

        desc_count += len(row_descriptions)
        descriptions = descriptions.union(row_descriptions)

    log.info('all descriptions variants num: %s' % len(descriptions))
    log.info('average description count per row: %.2f' % (desc_count / len(data)))

    # Save to files
    all_annotations = {'annotations': annotations}

    if not tf.gfile.Exists(args.output + '/images'):
        tf.gfile.MakeDirs(args.output + '/images')

    with open(os.path.join(args.output, 'annotations.json'), 'w') as f:
        f.write(json.dumps(all_annotations, indent=2))

    with open(os.path.join(args.output, 'label_map.json'), 'w') as f:
        f.write(json.dumps(label_map, indent=2, ensure_ascii=False))

    log.info('Written annotations and label map to %s' % args.output)

    file = tf.placeholder(tf.string, shape=None, name='file')
    image_data = tf.io.read_file(file)
    img = tf.image.decode_png(image_data, channels=3)
    img = tf.image.resize_bilinear([img], (299, 299))
    img = tf.cast(img, tf.float32)/127.5-1
    net = _inception(img)

    inception_variables_dict = {var.op.name: var for var in slim.get_model_variables('InceptionV3')}
    init_fn_inception = slim.assign_from_checkpoint_fn(args.inception_path, inception_variables_dict)

    with tf.Session() as sess:
        init_fn_inception(sess)
        for annot in annotations:
            res = sess.run(
                [net],
                {file: args.data_dir+'/images/'+annot['image_name']}
            )
            name = os.path.basename(f)
            np.save(args.output + '/images/' + name, res[0][0])
