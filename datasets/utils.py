import wget
import numpy as np
import pandas as pd
from tqdm import tqdm

from pathlib import Path
import zipfile
import libarchive
import tarfile
import sys


def download(url, savepath):
    wget.download(url, str(savepath))
    print()


def unzip(zippath, savepath):
    print("Extracting data...")
    zip = zipfile.ZipFile(zippath)
    zip.extractall(savepath)
    zip.close()


def unzip7z(zippath):
    print("Extracting data...")
    libarchive.extract_file(zippath)


def unziptargz(zippath, savepath):
    print("Extracting data...")
    f = tarfile.open(zippath)
    f.extractall(savepath)
    f.close()
