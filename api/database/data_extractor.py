from typing import Dict, Union

from pymongo import MongoClient


class NewsClassifierDB:
    """Class for connecting to MongoDB in order to save or get text data data

    Attributes
    ----------

    host : str
        Mongo's host

    port : int
        Mongo's port, by default 27017

    client : MongoClient
        Mongo's client
    """

    def __init__(self, host: str, port: int = 27017) -> None:
        """Method to initialize Mongo DB client

        Parameters
        ----------
        host : str
            Mongo's host
        port : int
            Mongo's port, by default 27017
        """

        self.host = host
        self.port = port
        self.client = MongoClient(f"mongodb://{self.host}:{self.port}/")

    def insert_prediction(
        self, prediction_data: Dict[str, Union[str, Dict[str, float]]]
    ):
        """Method to insert results of news classifier

        Parameters
        ----------
        prediction_data : Dict[str, Union[str, Dict[str, float]]]
            Data to insert into DB
        """
        self.client["news_data"]["classifier"].insert_one(prediction_data)
