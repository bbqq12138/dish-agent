"""
索引构建模块
"""

import logging
from typing import List
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class IndexConstructionModule:
    """索引构建模块 - 负责向量化和索引构建"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", index_save_path: str = "./chroma_vector_index", collection_name: str = 'dishes'):
        """
        初始化索引构建模块

        Args:
            model_name: 嵌入模型名称
            index_save_path: 索引保存路径
        """
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.collection_name = collection_name
        self.embeddings = None
        self.vectorstore = None
        self.setup_embeddings()
    
    def setup_embeddings(self):
        """初始化嵌入模型"""
        logger.info(f"正在初始化嵌入模型: {self.model_name}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        logger.info("嵌入模型初始化完成")
    
    def build_vector_index(self, chunks: List[Document], is_persist=True) -> Chroma:
        """
        构建向量索引
        
        Args:
            chunks: 文档块列表
            is_persist: 是否要持久化存储向量集合
            
        Returns:
            Chroma向量存储对象
        """
        logger.info("正在构建Chroma向量索引...")
        
        if not chunks:
            raise ValueError("文档块列表不能为空")
        
        # 构建Chroma向量集
        if is_persist:
            self.vectorstore = Chroma.from_documents(
                documents=chunks, 
                embedding=self.embeddings,
                collection_name=self.collection_name,  # 向量集合名，用于在同一目录下区分不同向量集
                persist_directory=self.index_save_path  # Chroma自动持久化存储
            )
            logger.info(f"向量索引构建完成，包含 {len(chunks)} 个向量")
            logger.info(f"向量索引已保存到: {self.index_save_path}")
        else:
            self.vectorstore = Chroma.from_documents(
                documents=chunks, 
                embedding=self.embeddings,
                collection_name=self.collection_name,  # 向量集合名，用于在同一目录下区分不同向量集
            )
            logger.info(f"向量索引构建完成，包含 {len(chunks)} 个向量")
            logger.info(f"向量索引没有持久化存储")
        
        return self.vectorstore
    
    def add_documents(self, new_chunks: List[Document]):
        """
        向现有索引添加新文档
        
        Args:
            new_chunks: 新的文档块列表
        """
        if not self.vectorstore:
            raise ValueError("请先构建向量索引")
        
        logger.info(f"正在添加 {len(new_chunks)} 个新文档到索引...")
        self.vectorstore.add_documents(new_chunks)
        logger.info("新文档添加完成")
    
    def load_index(self):
        """
        从配置的路径加载向量索引

        Returns:
            加载的向量存储对象，如果加载失败返回None
        """
        if not self.embeddings:
            self.setup_embeddings()

        if not Path(self.index_save_path).exists():
            logger.info(f"索引路径不存在: {self.index_save_path}，将构建新索引")
            return None

        vectorstore_temp = Chroma(
            collection_name=self.collection_name, 
            persist_directory=self.index_save_path, 
            embedding_function=self.embeddings, 
            collection_metadata={"hnsw:space": "cosine"},  # 使用余弦相似度，默认欧式距离
            create_collection_if_not_exists=False,
        )
        if vectorstore_temp._collection.count() > 0:
            self.vectorstore = vectorstore_temp
            logger.info(f"向量索引已从 {self.index_save_path} 加载")
            return self.vectorstore
        else:
            logger.warning(f"加载向量索引失败，向量数据库中不存在{self.collection_name}向量集，将构建新索引")
            return None
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            相似文档列表
        """
        if not self.vectorstore:
            raise ValueError("请先构建或加载向量索引")
        
        return self.vectorstore.similarity_search(query, k=k)


    # def save_index(self):
    #     """
    #     保存向量索引到配置的路径，Chroma没有这个功能，FAISS可以
    #     """
    #     if not self.vectorstore:
    #         raise ValueError("请先构建向量索引")

    #     # 确保保存目录存在，若不存在则将所有目录都创建
    #     Path(self.index_save_path).mkdir(parents=True, exist_ok=True)

    #     self.vectorstore.save_local(self.index_save_path)
        
    #     logger.info(f"向量索引已保存到: {self.index_save_path}")
