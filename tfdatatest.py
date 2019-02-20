#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import functools
import multiprocessing
import numpy as np
import os
import pandas
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from util.config import Config, initialize_globals
from util.text import text_to_char_array
from util.flags import create_flags, FLAGS
from timeit import default_timer as timer


tf.enable_eager_execution()


def read_csvs(csv_files):
    source_data = None
    for csv in csv_files:
        file = pandas.read_csv(csv, encoding='utf-8', na_filter=False)
        #FIXME: not cross-platform
        csv_dir = os.path.dirname(os.path.abspath(csv))
        file['wav_filename'] = file['wav_filename'].str.replace(r'(^[^/])', lambda m: os.path.join(csv_dir, m.group(1)))
        if source_data is None:
            source_data = file
        else:
            source_data = source_data.append(file)
    return source_data


def samples_to_mfccs(samples, sample_rate):
    spectrogram = contrib_audio.audio_spectrogram(samples, window_size=512, stride=320, magnitude_squared=True)
    mfccs = contrib_audio.mfcc(spectrogram, sample_rate, dct_coefficient_count=Config.n_input)
    mfccs = tf.reshape(mfccs, [-1, Config.n_input])

    return mfccs, tf.shape(mfccs)[0]


def file_to_features(wav_filename):
    samples = tf.read_file(wav_filename)
    decoded = contrib_audio.decode_wav(samples, desired_channels=1)
    features, features_len = samples_to_mfccs(decoded.audio, decoded.sample_rate)

    return features, features_len


def sparse_tuple_from(sequences, dtype=np.int32):
    r"""Creates a sparse representention of ``sequences``.
    Args:
        * sequences: a list of lists of type dtype where each element is a sequence
    Returns a tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), indices.max(0)[1]+1], dtype=np.int64)

    return tf.SparseTensor(indices=indices, values=values, dense_shape=shape)


def main(_):
    initialize_globals()

    print('Reading input files and processing transcripts...')
    df = read_csvs(FLAGS.train_files.split(','))
    df.sort_values(by='wav_filesize', inplace=True)
    df['transcript'] = df['transcript'].apply(functools.partial(text_to_char_array, alphabet=Config.alphabet))

    sparse = sparse_tuple_from(df['transcript'].values)

    t_set = (tf.data.Dataset.from_tensor_slices(sparse)
                            .batch(FLAGS.train_batch_size))

    f_set = (tf.data.Dataset.from_tensor_slices(df['wav_filename'].values)
                            .map(file_to_features, num_parallel_calls=multiprocessing.cpu_count())
                            .padded_batch(FLAGS.train_batch_size, padded_shapes=([None, Config.n_input], [])))

    z_set = tf.data.Dataset.zip((f_set, t_set))

    for (f, l), t in z_set:
        print(f)
        print(l)
        print(t)
        print()

    return

    def generate_values():
        for _, row in df.iterrows():
            yield tf.cast(row.wav_filename, tf.string), tf.cast(row.transcript, tf.int32)

    num_gpus = len(Config.available_devices)

    print('Creating input pipeline...')
    dataset = (tf.data.Dataset.from_generator(generate_values,
                                              output_types=(tf.string, tf.int32),
                                              output_shapes=([], [None]))
                              .map(file_to_features, num_parallel_calls=multiprocessing.cpu_count())
                              .prefetch(FLAGS.train_batch_size * num_gpus * 8)
                              .cache()
                              .padded_batch(FLAGS.train_batch_size,
                                            padded_shapes=([None, Config.n_input], [], [None]),
                                            drop_remainder=True)
                              .repeat(FLAGS.epoch)
              )

    batch_count = 0
    batch_size = None
    batch_time = 0

    start_time = timer()
    for batch_x, batch_x_len, batch_y in dataset:
        tf.print('batch x shape from iter: ', tf.shape(batch_x))
        batch_count += 1
        batch_size = batch_x.shape[0]
        print('.', end='')
    total_time = timer() - start_time
    print()
    print('Iterating through dataset took {:.3f}s, {} batches, {} epochs, batch size from dataset = {}, average batch time = {:.3f}'.format(total_time, batch_count, FLAGS.epoch, batch_size, batch_time/batch_count))


if __name__ == '__main__' :
    create_flags()
    tf.app.run(main)
