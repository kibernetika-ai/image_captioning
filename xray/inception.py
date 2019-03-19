import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import inception_v3


slim = tf.contrib.slim


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


def inception(inputs):
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
