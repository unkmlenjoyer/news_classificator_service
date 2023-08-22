"""Mongo's connector with defined methods to get / set / delete data (news)"""

from typing import Dict, List, Union

from pymongo import MongoClient, cursor
from pymongo.results import DeleteResult, InsertOneResult


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
    ) -> InsertOneResult:
        """Method to insert results of news classifier

        Parameters
        ----------
        prediction_data : Dict[str, Union[str, Dict[str, float]]]
            Data to insert into DB

        Returns
        -------
        InsertOneResult
        """
        return self.client["news_data"]["classifier"].insert_one(prediction_data)

    def select_news_short(self, news_ids: List[str]) -> cursor.Cursor:
        """Method to select only short representation of news

        Parameters
        ----------
        news_ids : List[str]
            ID's to select from database

        Returns
        -------
        pymongo.cursor.Cursor
            Iterator with result
        """
        return self.client["news_data"]["classifier"].find(
            filter={"text_id": {"$in": news_ids}},
            projection={"text_id": 1, "_id": 0, "insert_time": 1},
        )

    def get_one_news(self, news_id: str) -> Dict[str, Union[str, Dict[str, float]]]:
        """Get only one news item with specific identificator

        Parameters
        ----------
        news_id : str
            New's ID for text in database

        Returns
        -------
        Dict[str, Union[str, Dict[str, float]]]
            Data for only one news
        """
        return self.client["news_data"]["classifier"].find_one(
            {"text_id": news_id}, projection={"_id": 0}
        )

    def delete_one_news(self, news_id: str) -> DeleteResult:
        """Delete only one news

        Parameters
        ----------
        news_id : str
            New's ID for text in database

        Returns
        -------
        DeleteResult
        """

        return self.client["news_data"]["classifier"].delete_one({"text_id": news_id})
