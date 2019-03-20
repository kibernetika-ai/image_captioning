import json
import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.keras.backend import sparse_categorical_crossentropy
from tensorflow.python.training import session_run_hook

from xray import inception


const = tf.saved_model.signature_constants


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = sparse_categorical_crossentropy(real, pred, from_logits=True)
    # loss_ = tf.keras.losses.sparse_categorical_crossentropy(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def tokenize(word_index, text):
    tokens = [word_index['<start>']]

    for t in text.split(' '):
        v = word_index.get(t, None)
        if v is not None:
            tokens.append(v)

    tokens.append(word_index['<end>'])
    return tokens


def load_label_map(params):
    path = os.path.join(params['data_dir'], 'label_map.json')
    with open(path) as f:
        label_map = json.load(f)

    return label_map


def get_word_index(params):
    word_index = {}
    max_index = 0
    label_map = load_label_map(params)

    shift = 1
    for i, k in enumerate(sorted(label_map)):
        word_index[k] = i + shift
        max_index = i

    word_index['<end>'] = max_index + 1
    word_index['<start>'] = 0
    return word_index


def input_fn(params, is_training=True):
    # loading the numpy files
    if is_training:
        annot_path = os.path.join(params['data_dir'], 'train_annotations.json')
    else:
        annot_path = os.path.join(params['data_dir'], 'test_annotations.json')

    with open(annot_path) as f:
        annotations = json.load(f)

    batch_size = params['batch_size']
    word_index = params['word_index']
    end_token = word_index['<end>']

    def _input_fn():
        def generator():
            for annot in annotations['annotations']:

                img_path = os.path.join(params['data_dir'], 'images', annot['image_name'] + '.npy')
                if not os.path.exists(img_path):
                    continue

                caption = annot['caption']
                tokens = tokenize(word_index, caption)
                if len(tokens) > params['limit_length']:
                    tokens = tokens[:params['limit_length']]

                if len(tokens) < params['max_length']:
                    tokens.extend([word_index['<end>']] * (params['max_length'] - len(tokens)))

                img = np.load(img_path)
                img = np.reshape(img, (64, 2048))
                # img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                # img = img[:, :, ::-1]
                # img = cv2.resize(img, (299, 299))

                yield (img, np.array(tokens, dtype=np.int32))

        ds = tf.data.Dataset.from_generator(
            generator,
            (tf.float32, tf.int32),
            (tf.TensorShape([64, 2048]), tf.TensorShape([params['max_length']]))
        ).prefetch(100)

        # shuffling and batching
        if is_training:
            ds = ds.apply(tf.data.experimental.shuffle_and_repeat(100))

        ds = ds.padded_batch(
            batch_size,
            padded_shapes=([64, 2048], [params['max_length']]),
            padding_values=(0.0, np.int32(end_token))
        )
        # dataset = dataset.prefetch(buffer_size=batch_size * 5)

        return ds

    return _input_fn


def model_fn(features, labels, mode, params=None, config=None, model_dir=None):
    batch_size = params['batch_size']

    if mode == tf.estimator.ModeKeys.PREDICT:
        inputs = features['images']
        inputs = tf.cast(inputs, tf.float32) / 127.5 - 1
    else:
        inputs = tf.zeros((batch_size, 299, 299, 3))

    img = inception.inception(inputs)
    if mode == tf.estimator.ModeKeys.PREDICT:
        img = tf.reshape(img, (batch_size, 64, 2048))
    else:
        img = features
    # Reshape by inception outputs
    # img = tf.reshape(img, (-1, 64, 2048))

    encoder = CNN_Encoder(params['embedding_size'])
    decoder = RNN_Decoder(params['embedding_size'], params['units'], params['vocab_size'])

    loss = tf.constant(0, dtype=tf.float32)

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=batch_size)

    word_index = params['word_index']

    dec_input = tf.expand_dims([word_index['<start>']] * batch_size, 1)
    features = encoder(img)

    if mode in {tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL}:
        for i in range(0, labels.shape[1]):
            # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features=features, hidden=hidden)
            loss += loss_function(labels[:, i], predictions)
            # using teacher forcing
            dec_input = tf.expand_dims(labels[:, i], 1)

        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        if params.get('grad_clip') is None:
            train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())
        else:
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            if not params.get('train_inception'):
                grads = []
                vars = []
                for i, v in enumerate(variables):
                    if not v.name.startswith('InceptionV3'):
                        grads.append(gradients[i])
                        vars.append(variables[i])

                variables = vars
                gradients = grads

            gradients, _ = tf.clip_by_global_norm(gradients, params['grad_clip'])
            train_op = optimizer.apply_gradients(
                [(gradients[i], v) for i, v in enumerate(variables)],
                global_step=tf.train.get_or_create_global_step()
            )
        predictions = None
        export_outputs = None
    else:
        train_op = None
        max_length = params['max_length']
        attention_feature_shape = params['attention_features_shape']
        # attention_plot will be shaped [max_length, batch_size, feature_shape]
        attention_plot = None
        # ids will be shaped [max_length, batch_size]
        ids = None
        for i in range(max_length):
            predictions, hidden, attention_weights = decoder(dec_input, features=features, hidden=hidden)

            one_weight = tf.reshape(attention_weights, (1, batch_size, attention_feature_shape))
            if attention_plot is not None:
                attention_plot = tf.concat((attention_plot, one_weight), axis=0)
            else:
                attention_plot = tf.concat(one_weight, axis=0)

            predicted_id = tf.argmax(predictions, axis=1)
            predicted_ids = tf.reshape(predicted_id, (1, predicted_id.shape[0]))
            if ids is not None:
                ids = tf.concat((ids, predicted_ids), axis=0)
            else:
                ids = tf.concat(predicted_ids, axis=0)
            # result.append(tokenizer.index_word[predicted_id])

            # if word_index['<end>'] == predicted_id:
            #     break
            # if tokenizer.index_word[predicted_id] == '<end>':
            #     return ids, result, attention_plot

            dec_input = tf.expand_dims(predicted_id, 1)

        sig_def = const.DEFAULT_SERVING_SIGNATURE_DEF_KEY

        # transpose ids into [batch_size, max_length]
        ids = tf.transpose(ids)
        # transpose attention into [batch_size, max_length, feature_shape]
        attention_plot = tf.transpose(attention_plot, [1, 0, 2])

        predictions = {
            'predictions': ids,
            'attention': attention_plot,
        }
        export_outputs = {
            sig_def: tf.estimator.export.PredictOutput(predictions)
        }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        eval_metric_ops=None,
        predictions=predictions,
        loss=loss,
        export_outputs=export_outputs,
        train_op=train_op,
        training_hooks=[IniInceptionHook(params['inception_path'])],
    )
    # train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def __call__(self, features, **kwargs):
        hidden = kwargs.get('hidden')
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 64, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    # Since we have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def __call__(self, x, **kwargs):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, **kwargs):
        features = kwargs.get('features')
        hidden = kwargs.get('hidden')
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden=hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


class Model(tf.estimator.Estimator):
    def __init__(
            self,
            params=None,
            model_dir=None,
            config=None,
            warm_start_from=None
    ):
        def _model_fn(features, labels, mode, params, config):
            return model_fn(
                features=features,
                labels=labels,
                mode=mode,
                params=params,
                config=config)

        super(Model, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
            warm_start_from=warm_start_from
        )


class IniInceptionHook(session_run_hook.SessionRunHook):
    def __init__(self, model_path):
        self._model_path = model_path
        self._ops = None

    def begin(self):
        if self._model_path is not None:
            inception_variables_dict = {
                var.op.name: var
                for var in slim.get_model_variables('InceptionV3')
            }
            self._init_fn_inception = slim.assign_from_checkpoint_fn(self._model_path, inception_variables_dict)

    def after_create_session(self, session, coord):
        if self._model_path is not None:
            self._init_fn_inception(session)

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return None

    def after_run(self, run_context, run_values):
        None

