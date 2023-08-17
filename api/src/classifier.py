from typing import Dict, Union

import joblib
from sklearn.pipeline import Pipeline


class ArtifactLoader:
    """Class for loading artifacts (.pkl files)

    Returns
    -------
    Union[Pipeline, Dict[int, str]]
        Artifacts like scikit-learn Pipeline, Dicts for mapping
    """

    @staticmethod
    def load(path: str) -> Union[Pipeline, Dict[int, str]]:
        """Method to load artifact by path

        Parameters
        ----------
        path : str
            Path to artifact
            Must contains .pkl extension

        Returns
        -------
        Union[Pipeline, Dict[int, str]]
            Artifact
        """

        return joblib.load(path)
