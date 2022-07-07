from .base import AbstractDataset
from .utils import *

from datetime import date, datetime
from pathlib import Path
import pickle
import shutil
import tempfile
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


class Lastfm1kDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'lastfm1k'

    @classmethod
    def url(cls):
        return 'http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['README.txt',
                'userid-profile.tsv',
                'userid-timestamp-artid-artname-traid-traname.tsv']

    def maybe_download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if folder_path.is_dir() and\
           all(folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            print('Raw data already exists. Skip downloading')
            return
        
        print("Raw file doesn't exist. Downloading...")
        tmproot = Path(tempfile.mkdtemp())
        tmpzip = tmproot.joinpath('file.tar.gz')
        tmpfolder = tmproot.joinpath('folder')
        download(self.url(), tmpzip)
        unziptargz(tmpzip, tmpfolder)
        if self.zip_file_content_is_folder():
            tmpfolder = tmpfolder.joinpath(os.listdir(tmpfolder)[0])
        shutil.move(tmpfolder, folder_path)
        shutil.rmtree(tmproot)
        print()

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        self.maybe_download_raw_dataset()
        df = self.load_ratings_df()
        df = self.remove_immediate_repeats(df)
        df = self.filter_triplets(df)
        df, umap, smap = self.densify_index(df)
        train, val, test = self.split_df(df, len(umap))
        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'umap': umap,
                   'smap': smap}
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('userid-timestamp-artid-artname-traid-traname.tsv')
        df = pd.read_csv(file_path, sep='\t', header=None, usecols=[0, 1, 2])
        df.columns = ['uid', 'timestamp', 'sid']
        df = df.dropna()
        df['timestamp'] = df['timestamp'].map(
            lambda x: int(datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ").timestamp())
        )
        return df
