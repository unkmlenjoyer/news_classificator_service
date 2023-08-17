from motor.motor_asyncio import AsyncIOMotorClient


class MongoConnector:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.client = AsyncIOMotorClient()

    async def create_database(self):
        pass

    async def select_text(self):
        pass

    async def insert_prediction(self):
        pass
