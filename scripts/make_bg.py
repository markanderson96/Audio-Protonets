import pandas as pd
import numpy as np
import hydra
import os
from glob import glob
from omegaconf import DictConfig


@hydra.main(config_path='../config', config_name='config')
def main(conf: DictConfig):
    data_path = conf.path.data_dir
    train_dir = conf.path.train_dir
    val_dir = conf.path.val_dir

    csv_files = [file for path, _, _ in os.walk(conf.path.train_dir) 
                for file in glob(os.path.join(path, '*.csv')) ]

    for csv in csv_files:
        df = pd.read_csv(csv)
        classes = list(df)[3:]
        bg = (df[classes] == ('UNK' or 'NEG')).all(axis=1)
        bg = bg.replace(to_replace=True, value="POS")
        bg = bg.replace(to_replace=False, value="NEG")
        df["BG"] = bg
        df.to_csv(csv, index=None)

if __name__ == "__main__":
    main()