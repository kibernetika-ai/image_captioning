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

from xray import inception


slim = tf.contrib.slim
nltk.download('punkt')
tf.logging.set_verbosity(tf.logging.INFO)
log = tf.logging


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

    # Split up the annotations by groups
    groups = set()
    for annot in annotations:
        group = annot['image_id'].split('.')[0]
        groups.add(group)

    split_n = len(groups) // 10
    groups = list(groups)
    train_groups = set(groups[:-split_n])
    test_groups = set(groups[-split_n:])
    print('Train groups: %s ' % len(train_groups))
    print('Test groups: %s' % len(test_groups))

    train_annotations = []
    test_annotations = []
    for annot in annotations:
        group = annot['image_id'].split('.')[0]
        if group in train_groups:
            train_annotations.append(annot)
        elif group in test_groups:
            test_annotations.append(annot)

    # Save to files
    print()
    print('all annotations: %s' % len(annotations))
    print('train annotations: %s' % len(train_annotations))
    print('test annotations: %s' % len(test_annotations))
    all_annotations = {'annotations': annotations}

    if not tf.gfile.Exists(args.output + '/images'):
        tf.gfile.MakeDirs(args.output + '/images')

    with open(os.path.join(args.output, 'annotations.json'), 'w') as f:
        f.write(json.dumps(all_annotations, indent=2))

    with open(os.path.join(args.output, 'train_annotations.json'), 'w') as f:
        f.write(json.dumps({'annotations': train_annotations}, indent=2))

    with open(os.path.join(args.output, 'test_annotations.json'), 'w') as f:
        f.write(json.dumps({'annotations': test_annotations}, indent=2))

    with open(os.path.join(args.output, 'label_map.json'), 'w') as f:
        f.write(json.dumps(label_map, indent=2, ensure_ascii=False))

    log.info('Written annotations and label map to %s' % args.output)

    file = tf.placeholder(tf.string, shape=None, name='file')
    image_data = tf.io.read_file(file)
    img = tf.image.decode_png(image_data, channels=3)
    img = tf.image.resize_bilinear([img], (299, 299))
    img = tf.cast(img, tf.float32)/127.5-1
    net = inception.inception(img)

    inception_variables_dict = {var.op.name: var for var in slim.get_model_variables('InceptionV3')}
    init_fn_inception = slim.assign_from_checkpoint_fn(args.inception_path, inception_variables_dict)

    with tf.Session() as sess:
        init_fn_inception(sess)
        for annot in annotations:
            path = os.path.join(args.data_dir, 'images', annot['image_name'])
            if not os.path.exists(path):
                # Skip
                continue

            res = sess.run([net], {file: path})
            name = os.path.basename(annot['image_name'])
            np.save(args.output + '/images/' + name, res[0][0])
