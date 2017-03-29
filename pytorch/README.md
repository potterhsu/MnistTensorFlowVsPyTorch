## Usage

1. Convert to LMDB format

    ```
    $ python convert_to_lmdb.py --data_dir ../data
    ```

2. Test for reading LMDBs (Optional)

    Open `read_lmdb_sample.ipynb` in Jupyter

3. Train

    ```
    $ python train.py --data_dir ../data --logdir ./logs
    ```

4. Evaluate

    ```
    $ python eval.py --data_dir ../data ./logs/model-100.tar
    ```

5. Visualize

    ```
    $ python -m visdom.server
    $ python visualize.py --logdir ./logs
    ```

6. Try to make an inference (Optional)

    Open `inference_sample.ipynb` in Jupyter
