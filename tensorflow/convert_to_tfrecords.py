import os
import gzip
import numpy as np
import random
import tensorflow as tf
from meta import Meta

tf.app.flags.DEFINE_string('data_dir', '../data',
                           'Directory to read MNIST gzip files and write the converted files')
FLAGS = tf.app.flags.FLAGS


class Extractor(object):
    @staticmethod
    def _read32(bytestream):
        dt = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(bytestream.read(4), dtype=dt)[0]

    @staticmethod
    def extract_images(path_to_images_file):
        with open(path_to_images_file, 'rb') as f:
            print 'Extracting %s...' % f.name
            with gzip.GzipFile(fileobj=f) as bytestream:
                magic = Extractor._read32(bytestream)
                if magic != 2051:
                    raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, f.name))
                num_images = Extractor._read32(bytestream)
                rows = Extractor._read32(bytestream)
                cols = Extractor._read32(bytestream)
                buf = bytestream.read(rows * cols * num_images)
                images = np.frombuffer(buf, dtype=np.uint8)
                images = images.reshape(num_images, rows, cols, 1)
                return images

    @staticmethod
    def extract_labels(path_to_labels_file):
        with open(path_to_labels_file, 'rb') as f:
            print 'Extracting %s...' % f.name
            with gzip.GzipFile(fileobj=f) as bytestream:
                magic = Extractor._read32(bytestream)
                if magic != 2049:
                    raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, f.name))
                num_items = Extractor._read32(bytestream)
                buf = bytestream.read(num_items)
                labels = np.frombuffer(buf, dtype=np.uint8)
                return labels


class ExampleReader(object):
    def __init__(self, path_to_images_file, path_to_labels_file):
        self._images = Extractor.extract_images(path_to_images_file)
        self._labels = Extractor.extract_labels(path_to_labels_file)
        self._num_examples = self._images.shape[0]
        self._example_pointer = 0

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def read_and_convert(self):
        """
        Read and convert to example, returns None if no data is available.
        """
        if self._example_pointer == self._num_examples:
            return None
        image = self._images[self._example_pointer].tostring()
        label = int(self._labels[self._example_pointer])
        self._example_pointer += 1

        example = tf.train.Example(features=tf.train.Features(feature={
            'image': ExampleReader._bytes_feature(image),
            'label': ExampleReader._int64_feature(label)
        }))
        return example


def convert_to_tfrecords(path_to_images_file_and_labels_file_tuples,
                         path_to_tfrecords_files, choose_writer_callback):
    num_examples = []
    writers = []

    for path_to_tfrecords_file in path_to_tfrecords_files:
        num_examples.append(0)
        writers.append(tf.python_io.TFRecordWriter(path_to_tfrecords_file))

    for path_to_images_file, path_to_labels_file in path_to_images_file_and_labels_file_tuples:
        example_reader = ExampleReader(path_to_images_file, path_to_labels_file)
        while True:
            example = example_reader.read_and_convert()
            if example is None:
                break

            idx = choose_writer_callback(path_to_tfrecords_files)
            writers[idx].write(example.SerializeToString())
            num_examples[idx] += 1

    for writer in writers:
        writer.close()

    return num_examples


def create_tfrecords_meta_file(num_train_examples, num_val_examples, num_test_examples,
                               path_to_tfrecords_meta_file):
    print 'Saving meta file to %s...' % path_to_tfrecords_meta_file
    meta = Meta()
    meta.num_train_examples = num_train_examples
    meta.num_val_examples = num_val_examples
    meta.num_test_examples = num_test_examples
    meta.save(path_to_tfrecords_meta_file)


def main(_):
    path_to_train_images_file = os.path.join(FLAGS.data_dir, 'train-images-idx3-ubyte.gz')
    path_to_train_labels_file = os.path.join(FLAGS.data_dir, 'train-labels-idx1-ubyte.gz')
    path_to_test_images_file = os.path.join(FLAGS.data_dir, 't10k-images-idx3-ubyte.gz')
    path_to_test_labels_file = os.path.join(FLAGS.data_dir, 't10k-labels-idx1-ubyte.gz')

    path_to_train_tfrecords_file = os.path.join(FLAGS.data_dir, 'train.tfrecords')
    path_to_val_tfrecords_file = os.path.join(FLAGS.data_dir, 'val.tfrecords')
    path_to_test_tfrecords_file = os.path.join(FLAGS.data_dir, 'test.tfrecords')
    path_to_tfrecords_meta_file = os.path.join(FLAGS.data_dir, 'tfrecords_meta.json')

    for path_to_file in [path_to_train_tfrecords_file, path_to_val_tfrecords_file, path_to_test_tfrecords_file]:
        assert not os.path.exists(path_to_file), 'The file %s already exists' % path_to_file

    print 'Processing training and validation data...'
    [num_train_examples, num_val_examples] = convert_to_tfrecords([(path_to_train_images_file, path_to_train_labels_file)],
                                                                  [path_to_train_tfrecords_file, path_to_val_tfrecords_file],
                                                                  lambda paths: 0 if random.random() > 0.1 else 1)
    print 'Processing test data...'
    [num_test_examples] = convert_to_tfrecords([(path_to_test_images_file, path_to_test_labels_file)],
                                               [path_to_test_tfrecords_file],
                                               lambda paths: 0)

    create_tfrecords_meta_file(num_train_examples, num_val_examples, num_test_examples,
                               path_to_tfrecords_meta_file)

    print 'Done'


if __name__ == '__main__':
    tf.app.run(main=main)
