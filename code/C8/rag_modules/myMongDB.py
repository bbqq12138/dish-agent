from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

class MyMongoDB:
    def __init__(self) -> None:
        self._mongdb_client: AsyncMongoClient | None = None
        self.database: AsyncDatabase | None = None

    @classmethod
    async def create(cls, mongodb_uri: str, mongodb_database: str) -> 'MyMongoDB':
        """工厂方法创建MyMongoDB实例"""
        instance = cls()
        instance._mongdb_client = AsyncMongoClient(mongodb_uri)
        instance.database = await instance._inspect_database(mongodb_database)
        logger.info(f"✅ 成功连接到MongoDB数据库: {mongodb_database}")
        return instance


    async def insert_one(self, collection_name: str, document: dict | Document):
        if not await self.inspect_collection(collection_name):
            logger.warning(f"⚠️ 集合 '{collection_name}' 不存在，正在创建...")
            await self.create_collection(collection_name)

        collection = self.database.get_collection(collection_name)  # type: ignore
        if isinstance(document, Document):
            document = {
                '_id': document.metadata['parent_id'], 
                'dish_name': document.metadata['dish_name'], 
                'category': document.metadata['category'], 
                'difficulty': document.metadata['difficulty'], 
                'raw_content': document.page_content    # 原文本
            }
        result = await collection.insert_one(document)
        return result.inserted_id
    

    async def insert_many(self, collection_name: str, items: list[dict]):
        if not await self.inspect_collection(collection_name):
            logger.warning(f"⚠️ 集合 '{collection_name}' 不存在，正在创建...")
            await self.create_collection(collection_name)
            
        collection = self.database.get_collection(collection_name)  # type: ignore
        docs_to_insert = [item for item in items if isinstance(item, dict)]

        result = await collection.insert_many(docs_to_insert)

        logger.info(f"✅ 成功插入 {len(result.inserted_ids)} 条文档到集合 '{collection_name}'")
        return result.inserted_ids

    
    async def find_one(self, collection_name: str, *query: dict):
        """
        在指定集合中查找单个文档

        Args:
            collection_name: 集合名称
            *query: 查询条件，MongoDB查询语法
            eg.query={'category': '早餐'}, {'dish_name': 1, 'category': 1}
        """
        if not await self.inspect_collection(collection_name):
            logger.warning(f"⚠️ 集合 '{collection_name}' 不存在！")
            return None
        
        collection = self.database.get_collection(collection_name)  # type: ignore
        result = await collection.find_one(*query)
        return result


    async def find_many(self, collection_name: str, *query: dict):
        """
        在指定集合中查找多个文档

        Args:
            collection_name: 集合名称
            *query: 查询条件，MongoDB查询语法
            eg.query={'category': '早餐'}, {'dish_name': 1, 'category': 1}  # 查询过滤条件和返回字段
        """
        if not await self.inspect_collection(collection_name):
            logger.warning(f"⚠️ 集合 '{collection_name}' 不存在！")
            return []
        
        collection = self.database.get_collection(collection_name)  # type: ignore
        cursor = collection.find(*query)
        results = await cursor.to_list(length=None)
        return results


    async def inspect_collection(self, collection_name: str) -> bool:
        """检查集合是否存在，如果不存在则创建"""
        collections = await self.database.list_collection_names()  # type: ignore
        return collection_name in collections
    

    async def _inspect_database(self, database_name: str) -> AsyncDatabase:
        """检查数据库是否存在"""
        databases = await self._mongdb_client.list_database_names()  # type: ignore
        if database_name not in databases:
            logger.warning(f"⚠️ 数据库 '{database_name}' 不存在！")
        return self._mongdb_client.get_database(database_name)  # type: ignore


    async def get_existing_collections(self):
        """获取现有集合列表"""
        return await self.database.list_collection_names()  # type: ignore

    async def create_collection(self, collection_name: str):
        """创建集合，如果集合已存在则忽略"""
        existing_collections = await self.database.list_collection_names()  # type: ignore
        if collection_name not in existing_collections:
            await self.database.create_collection(collection_name)  # type: ignore
            print(f"✅ 集合 '{collection_name}' 创建成功！")
        else:
            print(f"⚠️ 集合 '{collection_name}' 已存在，跳过创建。")

    
    async def get_collection_count(self, collection_name: str) -> int:
        """获取集合中的文档数量"""
        if not await self.inspect_collection(collection_name):
            logger.warning(f"⚠️ 集合 '{collection_name}' 不存在！")
            return 0
        
        collection = self.database.get_collection(collection_name)  # type: ignore
        count = await collection.count_documents({})
        return count

    async def distinct_values(self, collection_name: str, field_name: str) -> list:
        """获取集合中指定字段的不同值列表"""
        if not await self.inspect_collection(collection_name):
            logger.warning(f"⚠️ 集合 '{collection_name}' 不存在！")
            return []
        
        collection = self.database.get_collection(collection_name)  # type: ignore
        distinct_values = await collection.distinct(field_name)
        return distinct_values