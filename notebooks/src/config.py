"""Class with research configuration"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ClassifierResearchConfig:
    """Config for new's classifier research

    Attributes
    ----------

    RAW_DATA_PATH : str
        Path to raw data folder

    CLEAN_DATA_PATH : str
        Path to cleaned data folder

    ARTIFACT_PATH : str
        Path to artifact folder

    PD_MAX_ROWS : int
        Amount of visible rows in pandas output

    PD_MAX_COLS : int
        Amount of visible columns in pandas output

    RANDOM_SEED : int
        Fixed seed to reproduce research
    """

    RAW_DATA_PATH: str = "../data/raw/lenta-ru-news.csv"
    CLEAN_DATA_PATH: str = "../data/processed/"
    ARTIFACT_PATH: str = "../artifacts/"

    PD_MAX_ROWS: int = 100
    PD_MAX_COLS: int = 50
    SNS_STYLE: str = "darkgrid"
    SNS_FIG_SIZE: Tuple[int, int] = (40, 30)
    RANDOM_SEED: int = 42
