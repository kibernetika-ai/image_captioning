import json
import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.training import session_run_hook

from xray import inception


const = tf.saved_model.signature_constants


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = tf.keras.losses.sparse_categorical_crossentropy(real, pred)

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


def get_word_index(params):
    word_index = {}
    max_index = 0
    path = os.path.join(params['data_dir'], 'label_map.json')
    with open(path) as f:
        label_map = json.load(f)

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

                img_path = os.path.join(params['data_dir'], 'images', annot['image_name'])
                if not os.path.exists(img_path):
                    continue

                caption = annot['caption']
                tokens = tokenize(word_index, caption)
                if len(tokens) > params['limit_length']:
                    tokens = tokens[:params['limit_length']]

                if len(tokens) < params['max_length']:
                    tokens.extend([word_index['<end>']] * (params['max_length'] - len(tokens)))

                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = img[:, :, ::-1]
                img = cv2.resize(img, (299, 299))

                yield (img, np.array(tokens, dtype=np.int32))

        ds = tf.data.Dataset.from_generator(
            generator,
            (tf.float32, tf.int32),
            (tf.TensorShape([299, 299, 3]), tf.TensorShape([params['max_length']]))
        )

        # shuffling and batching
        if is_training:
            ds = ds.apply(tf.data.experimental.shuffle_and_repeat(100))

        ds = ds.padded_batch(
            batch_size,
            padded_shapes=([299, 299, 3], [params['max_length']]),
            padding_values=(0.0, np.int32(end_token))
        )
        # dataset = dataset.prefetch(buffer_size=batch_size * 5)

        return ds

    return _input_fn


def model_fn(features, labels, mode, params=None, config=None, model_dir=None):
    inputs = features
    if isinstance(features, dict):
        inputs = features['images']

    img = tf.cast(inputs, tf.float32) / 127.5 - 1
    img = inception.inception(img)
    # Reshape by inception outputs
    img = tf.reshape(img, (-1, 64, 2048))

    encoder = CNN_Encoder(params['embedding_size'])
    decoder = RNN_Decoder(params['embedding_size'], params['units'], params['vocab_size'])

    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])

    loss = tf.constant(0, dtype=tf.float32)

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=params['batch_size'])

    word_index = params['word_index']

    dec_input = tf.expand_dims([word_index['<start>']] * params['batch_size'], 1)

    if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.GradientTape() as tape:
            features = encoder(img)

            for i in range(1, labels.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = decoder(dec_input, features=features, hidden=hidden)
                loss += loss_function(labels[:, i], predictions)
                # using teacher forcing
                dec_input = tf.expand_dims(labels[:, i], 1)

        total_loss = (loss / int(labels.shape[1]))
        trainable_variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        train_op = optimizer.apply_gradients(
            zip(gradients, trainable_variables),
            global_step=tf.train.get_or_create_global_step()
        )
        predictions = None
        export_outputs = None
    else:
        total_loss = None
        train_op = None
        max_length = params['max_length']
        attention_plot = np.zeros((max_length, params['attention_features_shape']))
        ids = tf.zeros((max_length,), dtype=tf.int32)
        for i in range(max_length):
            predictions, hidden, attention_weights = decoder(dec_input, features=features, hidden=hidden)

            attention_plot[i] = tf.reshape(attention_weights, (-1,))

            predicted_id = tf.argmax(predictions[0])
            ids[i] = predicted_id
            # result.append(tokenizer.index_word[predicted_id])

            if word_index['<end>'] == predicted_id:
                break
            # if tokenizer.index_word[predicted_id] == '<end>':
            #     return ids, result, attention_plot

            dec_input = tf.expand_dims([predicted_id], 0)

        sig_def = const.DEFAULT_SERVING_SIGNATURE_DEF_KEY
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
        loss=total_loss,
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
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
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

