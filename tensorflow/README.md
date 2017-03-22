## Usage

1. Convert to TFRecords format

    ```
    $ python convert_to_tfrecords.py --data_dir ../data
    ```

2. Test for reading TFRecords files (Optional)

    Open `read_tfrecords_sample.ipynb` in Jupyter

3. Train

    ```
    $ python train.py --data_dir ../data --train_logdir ./logs/train
    ```

4. Evaluate

    ```
    $ python eval.py --data_dir ../data --ckeckpoint_dir ./logs/train --eval_logdir ./logs/eval
    ```

5. Visualize

    ```
    $ tensorboard --logdir ./logs
    ```

6. Try to make an inference (Optional)

    Open `inference_sample.ipynb` in Jupyter
