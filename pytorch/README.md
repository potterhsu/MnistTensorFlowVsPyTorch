## Usage

1. Convert to LMDB format

    ```
    $ python convert_to_lmdb.py --data_dir ../data
    ```

2. Train

    ```
    $ python train.py --data_dir ../data --train_logdir ./logs/train
    ```

3. Evaluate

    ```
    $ python eval.py --data_dir ../data ./logs/train/model-100.tar
    ```

