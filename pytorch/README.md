## Requirements

* Python 2.7
* PyTorch
* Protocol Buffers 3
* LMDB
* Visdom

## Usage

1. Convert to LMDB format

    ```
    $ python convert_to_lmdb.py --data_dir ../data
    ```

1. (Optional) Test for reading LMDBs

    ```
    Open `read_lmdb_sample.ipynb` in Jupyter
    ```

1. Train

    ```
    $ python train.py --data_dir ../data --logdir ./logs
    ```

1. Retrain if you need

    ```
    $ python train.py --data_dir ./data --logdir ./logs_retrain --restore_checkpoint ./logs/model-100.tar
    ```

1. Evaluate

    ```
    $ python eval.py --data_dir ../data ./logs/model-100.tar
    ```

1. Visualize

    ```
    $ python -m visdom.server
    $ python visualize.py --logdir ./logs
    ```

1. (Optional) Try to make an inference

    ```
    Open `inference_sample.ipynb` in Jupyter
    ```
    
1. Clean

    ```
    $ rm -rf ./logs
    or
    $ rm -rf ./logs_retrain
    ```
