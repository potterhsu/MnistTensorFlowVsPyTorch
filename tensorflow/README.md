## Usage

1. Convert to TFRecords format

    ```
    $ python convert_to_tfrecords.py --data_dir ../data
    ```

2. Train

    ```
    $ python train.py --data_dir ../data --train_logdir ./logs/train
    ```

3. Evaluate

    ```
    $ python eval.py --data_dir ../data --ckeckpoint_dir ./logs/train --eval_logdir ./logs/eval
    ```

4. Visualize

    ```
    $ tensorboard --logdir ./logs
    ```
