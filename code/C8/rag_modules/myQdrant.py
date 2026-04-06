from qdrant_client import AsyncQdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from rag_modules import JiebaFastEmbed
from FlagEmbedding import FlagReranker


class Qdrant(AsyncQdrantClient):
    """
    Qdrant 向量数据库客户端封装类，继承自 AsyncQdrantClient
    主要目的是为了在需要时添加一些自定义的方法或属性
    
    目前主要添加了对稠密嵌入模型、稀疏嵌入模型和重排序模型的支持，可以在初始化时设置默认模型，或者在使用时动态指定
     - dense_model: HuggingFaceEmbeddings 类型的稠密嵌入模型
     - sparse_model: JiebaFastEmbed 类型的稀疏嵌入模型
     - reranker: FlagReranker 类型的重排序模型
    """

    def __init__(self, url: str, **kwargs):
        """
        初始化 Qdrant 客户端
        
        Args:
            url: Qdrant 服务器 URL
            **kwargs: 其他传递给 AsyncQdrantClient 的参数
        """
        super().__init__(url=url, **kwargs)
        self.dense_model: HuggingFaceEmbeddings | None = None  # 可以在初始化时设置默认的稠密嵌入模型，或者在使用时动态指定
        self.sparse_model: JiebaFastEmbed | None = None  # 可以在初始化时设置默认的稀疏嵌入模型，或者在使用时动态指定
        self.reranker: FlagReranker | None = None  # 可以在初始化时设置默认的重排序模型，或者在使用时动态指定

    def set_dense_model(self, dense_model: HuggingFaceEmbeddings):
        """设置稠密嵌入模型"""
        self.dense_model = dense_model

    def set_sparse_model(self, sparse_model: JiebaFastEmbed):
        """设置稀疏嵌入模型"""
        self.sparse_model = sparse_model

    def set_reranker(self, reranker: FlagReranker):
        """设置重排序模型"""
        self.reranker = reranker
