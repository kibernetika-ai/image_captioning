from __future__ import absolute_import, division, print_function

import argparse
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

from xray import model

slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.INFO)
log = tf.logging


def calc_max_length(tensor):
    return max(len(t) for t in tensor)


def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result, len_result, l + 1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()


# captions on the validation set
# rid = np.random.randint(0, len(img_name_val))
# image = img_name_val[rid]
# real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
# ids, result, attention_plot = evaluate(image)
# print(cap_val[rid])
# print('Real Caption:', real_caption)
# for real in real_caption.split()[1:-1]:
#     print('  %s' % label_map[real])
#
# print(ids)
# print('Pred Caption:        ', ' '.join(result))
# for pred in result[:-1]:
#     print('  %s' % label_map[pred])
#
# plot_attention(image, result, attention_plot)
# opening the image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', type=str, default='./annotations.json', help='Annotations file path')
    parser.add_argument('--data-dir', type=str, default='./data', help='Dataset directory')
    parser.add_argument('--train-dir', type=str, default='training', help='Training dir')
    parser.add_argument('--inception-path', type=str, default='./inception_v3.ckpt', help='Inception checkpoint')
    parser.add_argument('--steps', type=int, default=1000, help='Training steps count')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--log-step-count-steps', type=int, default=5, help='Log every N step')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--mode', default='train', choices=['train', 'export', 'eval'], help='Mode')
    parser.add_argument('--eval', default=False, action='store_true', help='Run evaluation during train')
    parser.add_argument('--export', default=False, action='store_true', help='Changes mode to export')
    return parser.parse_args()


def export(xray, train_dir, params):
    feature_placeholders = {
        'images': tf.placeholder(tf.float32, [params['batch_size'], 299, 299, 3], name='images'),
    }
    receiver = tf.estimator.export.build_raw_serving_input_receiver_fn(
        feature_placeholders,
        default_batch_size=params['batch_size']
    )
    export_path = xray.export_savedmodel(
        train_dir,
        receiver,
    )
    export_path = export_path.decode("utf-8")
    log.info('Exported to %s.' % export_path)

    shutil.copy(
        os.path.join(params['data_dir'], 'label_map.json'),
        os.path.join(export_path, 'label_map.json'),
    )


def main():
    args = parse_args()
    params = {
        'batch_size': args.batch_size,
        'buffer_size': 1000,
        'embedding_size': 256,
        'units': 512,
        'limit_length': 10,
        'grad_clip': 1.0,
        'learning_rate': args.learning_rate,
        'data_dir': args.data_dir,
        'inception_path': args.inception_path,
        'vocab_size': 0,
        'attention_features_shape': 64,
        'features_shape': 2048,
        'log_step_count_steps': args.log_step_count_steps,
        'keep_checkpoint_max': 5,
    }

    params['word_index'] = model.get_word_index(params)
    params['max_length'] = params['limit_length']
    vocab_size = len(params['word_index'])
    params['vocab_size'] = vocab_size

    conf = tf.estimator.RunConfig(
        model_dir=args.train_dir,
        save_summary_steps=100,
        save_checkpoints_secs=120,
        save_checkpoints_steps=None,
        keep_checkpoint_max=params['keep_checkpoint_max'],
        log_step_count_steps=params['log_step_count_steps'],
    )
    xray = model.Model(
        params=params,
        model_dir=args.train_dir,
        config=conf,
    )
    mode = args.mode
    if args.export:
        mode = 'export'

    if mode == 'train':
        input_fn = model.input_fn(params, True)

        if args.eval:
            eval_input_fn = model.input_fn(params, False)
            train_spec = tf.estimator.TrainSpec(input_fn=input_fn, max_steps=args.steps)
            eval_spec = tf.estimator.EvalSpec(
                input_fn=eval_input_fn, steps=1, start_delay_secs=10, throttle_secs=10
            )
            tf.estimator.train_and_evaluate(xray, train_spec, eval_spec)
        else:
            xray.train(input_fn=input_fn, steps=args.steps)
    elif mode == 'eval':
        eval_input_fn = model.input_fn(params, False)
        xray.evaluate(eval_input_fn, steps=1)
    elif mode == 'export':
        # export
        export(xray, args.train_dir, params)


if __name__ == '__main__':
    main()
