"""
索引构建模块
"""

import logging
from typing import List
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from qdrant_client import models
from .jieba_fastembed import JiebaFastEmbed
from qdrant_client.models import PointStruct, SparseVector
from .myQdrant import Qdrant


logger = logging.getLogger(__name__)

class IndexConstructionModule:
    """索引构建模块 - 负责向量化和索引构建"""

    def __init__(self, dense_model_name: str = "BAAI/bge-large-zh-v1.5", collection_name: str = 'RecipeChunk', qdrant_url: str = "http://localhost:6334"):
        """
        初始化索引构建模块

        Args:
            dense_model_name: 稠密嵌入模型名称
            collection_name: 向量数据库集合名称
            qdrant_url: 向量数据库URL
        """
        self.collection_name = collection_name
        self.qdrant_client: Qdrant | None = None
        self._setup(dense_model_name, qdrant_url)
    

    def _setup(self, dense_model_name: str, qdrant_url: str):
        """初始化稀疏、稠密嵌入模型、向量数据库客户端等资源"""

        logger.info("正在初始化向量数据库客户端...")
        self.qdrant_client = Qdrant(
            url=qdrant_url, 
            check_compatibility=False, 
            prefer_grpc=True,
        )
        if not self.qdrant_client:
            logger.error("向量数据库客户端初始化失败")
            raise ValueError("向量数据库客户端初始化失败")
        logger.info("向量数据库客户端初始化完成")


        logger.info(f"正在初始化稠密模型: {dense_model_name} 和稀疏模型: BM25...")
        dense_model = HuggingFaceEmbeddings(
            model_name=dense_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        sparse_model = JiebaFastEmbed(model_name='Qdrant/bm25')  # 重新封装的适合中文的稀疏向量模型
        self.qdrant_client.set_dense_model(dense_model)
        self.qdrant_client.set_sparse_model(sparse_model)
        logger.info("嵌入模型初始化完成")

    

    async def build_vector_index(self, chunks: List[Document]):
        """
        构建向量索引
        
        Args:
            chunks: 文档块列表
            vectorstore_url: 向量数据库URL
            
        Returns:
            AsyncQdrantClient
        """
        logger.info("正在构建Qdrant向量索引...")
        
        if not chunks:
            raise ValueError("文档块列表不能为空")
        
        if not self.qdrant_client:
            raise ValueError("向量数据库客户端未初始化")
        
        # 检查集合是否存在，不存在则创建
        if await self.qdrant_client.collection_exists(collection_name=self.collection_name):
            logger.info(f"集合 '{self.collection_name}' 已存在，跳过集合创建步骤")
        else:
            logger.info(f"集合 '{self.collection_name}' 不存在. 正在创建集合...")
            await self.qdrant_client.create_collection(
                collection_name=self.collection_name, 
                vectors_config={'dense': models.VectorParams(
                    size=1024, 
                    distance=models.Distance.COSINE, 
                    on_disk=True,  # 稠密向量是否在内存，可以节省内存，牺牲时间
                    datatype=models.Datatype.FLOAT32,  # 默认32,16节省内存
                )}, 
                sparse_vectors_config={'sparse': models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=True,  # 内存优化：稀疏索引放硬盘
                        datatype=models.Datatype.FLOAT32
                    )
                )}, 
                hnsw_config=models.models.HnswConfigDiff(on_disk=True), 
                on_disk_payload=False  # 元数据直接存在内存中，元数据不大
            )
            logger.info(f"集合 '{self.collection_name}' 创建完成")
        
        logger.info("正在构建并存储向量索引...")
        await self._store_chunks_in_qdrant(chunks)
        logger.info(f"集合 '{self.collection_name}' 的向量索引构建完成")
    

    async def _store_chunks_in_qdrant(self, chunks: List[Document]):
        """将文档块存储到Qdrant中，使用稠密和稀疏向量"""
        if not self.qdrant_client:
            raise ValueError("请先构建向量索引")
        
        # 构建稠密、稀疏向量
        logger.info("正在构建稠密和稀疏向量...")
        chunk_texts = [chunk.page_content for chunk in chunks]
        if self.qdrant_client.dense_model is None or self.qdrant_client.sparse_model is None:
            raise ValueError("嵌入模型未初始化")
        dense_vecs = await self.qdrant_client.dense_model.aembed_documents(chunk_texts)
        sparse_vecs = self.qdrant_client.sparse_model.embed_batch(chunk_texts)
        logger.info(f"已构建 {len(dense_vecs)} 个稠密向量和 {len(sparse_vecs)} 个稀疏向量.")
        
        # 构建数据点对象
        logger.info("正在构建Qdrant数据点对象...")
        points = []
        for dense_vec, sparse_vec, chunk in zip(dense_vecs, sparse_vecs, chunks):
            point = PointStruct(
                id = chunk.metadata['chunk_id'], 
                vector={
                    'dense': dense_vec, 
                    'sparse': SparseVector(
                        indices=sparse_vec.indices.tolist(), # toekn对应的索引
                        values=sparse_vec.values.tolist(),   # 索引对应的token的得分
                    )
                }, 
                payload=chunk.metadata
            )
            points.append(point)
        logger.info(f"已构建 {len(points)} 个数据点对象.")
            
        # 存储数据点到Qdrant
        logger.info(f"正在将 {len(points)} 个数据点插入到 Qdrant 集合 '{self.collection_name}' 中...")
        await self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"已将 {len(points)} 个数据点插入到 Qdrant 集合 '{self.collection_name}' 中.")

    
    async def add_documents(self, new_chunks: List[Document]):
        """
        向现有索引添加新文档
        
        Args:
            new_chunks: 新的文档块列表
        """
        if not self.qdrant_client:
            raise ValueError("请先构建向量索引")
        
        logger.info(f"正在添加 {len(new_chunks)} 个新文档到索引...")
        await self._store_chunks_in_qdrant(new_chunks)
        logger.info("新文档添加完成")
    

    async def load_index(self) -> bool:
        """
        从配置的路径加载向量索引

        Returns:
            加载的向量存储对象，如果加载失败返回None
        """

        # 检查集合是否存在，不存在则创建
        if self.qdrant_client is None:
            logger.error("向量数据库客户端未初始化，无法加载索引")
            return False
        
        if not await self.qdrant_client.collection_exists(collection_name=self.collection_name):
            logger.warning(f"集合 '{self.collection_name}' 不存在，将重新构建集合与向量索引")
            return False

        
        if (await self.qdrant_client.count(collection_name=self.collection_name)).count > 0:
            logger.info(f"集合 '{self.collection_name}' 中已有数据，加载向量索引成功")
            return True
        else:
            logger.warning(f"加载向量索引失败，向量数据库中不存在{self.collection_name}向量集，将构建新索引")
            return False