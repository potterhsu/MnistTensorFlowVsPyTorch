import tensorflow as tf
from donkey import Donkey
from model import Model


class Evaluator(object):
    def __init__(self, path_to_eval_log_dir):
        self.summary_writer = tf.summary.FileWriter(path_to_eval_log_dir)

    def evaluate(self, path_to_checkpoint, path_to_tfrecords_file, num_examples, global_step):
        batch_size = 128
        num_batches = num_examples / batch_size

        with tf.Graph().as_default():
            images, labels = Donkey.build_batch(path_to_tfrecords_file, batch_size=batch_size,
                                                one_hot=False, shuffled=False)
            logits = Model.inference(images, keep_prob=1)
            predictions = tf.argmax(logits, axis=1)

            accuracy, update_accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)

            tf.summary.image('image', images)
            tf.summary.scalar('accuracy', accuracy)
            summary = tf.summary.merge_all()

            with tf.Session() as sess:
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                restorer = tf.train.Saver()
                restorer.restore(sess, path_to_checkpoint)

                for _ in xrange(num_batches):
                    sess.run(update_accuracy)

                accuracy_val, summary_val = sess.run([accuracy, summary])
                self.summary_writer.add_summary(summary_val, global_step=global_step)

                coord.request_stop()
                coord.join(threads)

        return accuracy_val
