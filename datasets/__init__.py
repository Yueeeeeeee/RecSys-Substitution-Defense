from .lastfm1k import Lastfm1kDataset
from .ml_1m import ML1MDataset
from .ml_20m import ML20MDataset
from .games import GamesDataset
from .steam import SteamDataset
from .beauty import BeautyDataset
from .yoochoose import YooChooseDataset

DATASETS = {
    ML1MDataset.code(): ML1MDataset,
    Lastfm1kDataset.code(): Lastfm1kDataset,
    ML20MDataset.code(): ML20MDataset,
    SteamDataset.code(): SteamDataset,
    GamesDataset.code(): GamesDataset,
    BeautyDataset.code(): BeautyDataset,
    YooChooseDataset.code(): YooChooseDataset
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
