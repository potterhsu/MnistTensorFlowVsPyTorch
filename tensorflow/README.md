## Requirements

* Python 2.7
* Tensorflow

## Usage

1. Convert to TFRecords format

    ```
    $ python convert_to_tfrecords.py --data_dir ../data
    ```

1. (Optional) Test for reading TFRecords files

    Open `read_tfrecords_sample.ipynb` in Jupyter

1. Train

    ```
    $ python train.py --data_dir ../data --train_logdir ./logs/train
    ```
    
1. Retrain if you need
    ```
    $ python train.py --data_dir ./data --train_logdir ./logs/train2 --restore_checkpoint ./logs/train/latest.ckpt
    ```

1. Evaluate

    ```
    $ python eval.py --data_dir ../data --ckeckpoint_dir ./logs/train --eval_logdir ./logs/eval
    ```

1. Visualize

    ```
    $ tensorboard --logdir ./logs
    ```

1. (Optional) Try to make an inference

    Open `inference_sample.ipynb` in Jupyter

1. Clean

    ```
    $ rm -rf ./logs
    or
    $ rm -rf ./logs/train2
    or
    $ rm -rf ./logs/eval
    ```
