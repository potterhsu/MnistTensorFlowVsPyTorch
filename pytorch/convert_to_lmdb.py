import argparse
import os
import gzip
import numpy
import random
import lmdb
import example_pb2
from meta import Meta

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', default='../data', help='directory to read MNIST gzip files and write the converted files')


class Extractor(object):
    @staticmethod
    def _read32(bytestream):
        dt = numpy.dtype(numpy.uint32).newbyteorder('>')
        return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

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
                images = numpy.frombuffer(buf, dtype=numpy.uint8)
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
                labels = numpy.frombuffer(buf, dtype=numpy.uint8)
                return labels


class ExampleReader(object):
    def __init__(self, path_to_images_file, path_to_labels_file):
        self._images = Extractor.extract_images(path_to_images_file)
        self._labels = Extractor.extract_labels(path_to_labels_file)
        self._num_examples = self._images.shape[0]
        self._example_pointer = 0

    def read_and_convert(self):
        """
        Read and convert to example, returns None if no data is available.
        """
        if self._example_pointer == self._num_examples:
            return None
        image = self._images[self._example_pointer]
        label = int(self._labels[self._example_pointer])
        self._example_pointer += 1

        example = example_pb2.Example()
        example.image = image.tostring()
        example.label = label
        return example


def convert_to_lmdb(path_to_images_file_and_labels_file_tuples,
                    path_to_lmdb_dirs, choose_writer_callback):
    num_examples = []
    writers = []

    for path_to_lmdb_dir in path_to_lmdb_dirs:
        num_examples.append(0)
        writers.append(lmdb.open(path_to_lmdb_dir, map_size=10*1024*1024*1024))

    for path_to_images_file, path_to_labels_file in path_to_images_file_and_labels_file_tuples:
        example_reader = ExampleReader(path_to_images_file, path_to_labels_file)
        while True:
            example = example_reader.read_and_convert()
            if example is None:
                break

            idx = choose_writer_callback(path_to_lmdb_dirs)
            with writers[idx].begin(write=True) as txn:
                str_id = '{:08}'.format(num_examples[idx] + 1)
                txn.put(str_id, example.SerializeToString())
            num_examples[idx] += 1

    for writer in writers:
        writer.close()

    return num_examples


def create_lmdb_meta_file(num_train_examples, num_val_examples, num_test_examples, path_to_lmdb_meta_file):
    print 'Saving meta file to %s...' % path_to_lmdb_meta_file
    meta = Meta()
    meta.num_train_examples = num_train_examples
    meta.num_val_examples = num_val_examples
    meta.num_test_examples = num_test_examples
    meta.save(path_to_lmdb_meta_file)


def main(args):
    path_to_train_images_file = os.path.join(args.data_dir, 'train-images-idx3-ubyte.gz')
    path_to_train_labels_file = os.path.join(args.data_dir, 'train-labels-idx1-ubyte.gz')
    path_to_test_images_file = os.path.join(args.data_dir, 't10k-images-idx3-ubyte.gz')
    path_to_test_labels_file = os.path.join(args.data_dir, 't10k-labels-idx1-ubyte.gz')

    path_to_train_lmdb_dir = os.path.join(args.data_dir, 'train.lmdb')
    path_to_val_lmdb_dir = os.path.join(args.data_dir, 'val.lmdb')
    path_to_test_lmdb_dir = os.path.join(args.data_dir, 'test.lmdb')
    path_to_lmdb_meta_file = os.path.join(args.data_dir, 'lmdb_meta.json')

    for path_to_dir in [path_to_train_lmdb_dir, path_to_val_lmdb_dir, path_to_test_lmdb_dir]:
        assert not os.path.exists(path_to_dir), 'LMDB directory %s already exists' % path_to_dir

    print 'Processing training and validation data...'
    [num_train_examples, num_val_examples] = convert_to_lmdb([(path_to_train_images_file, path_to_train_labels_file)],
                                                             [path_to_train_lmdb_dir, path_to_val_lmdb_dir],
                                                             lambda paths: 0 if random.random() > 0.1 else 1)
    print 'Processing test data...'
    [num_test_examples] = convert_to_lmdb([(path_to_test_images_file, path_to_test_labels_file)],
                                          [path_to_test_lmdb_dir],
                                          lambda paths: 0)

    create_lmdb_meta_file(num_train_examples, num_val_examples, num_test_examples, path_to_lmdb_meta_file)

    print 'Done'


if __name__ == '__main__':
    main(parser.parse_args())
