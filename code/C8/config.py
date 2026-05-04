"""
RAG系统配置文件
"""

from dataclasses import dataclass, field
from dotenv import load_dotenv
import os
from typing import Dict, Any

load_dotenv()


@dataclass
class RAGConfig:
    """RAG系统配置类"""

    # 路径配置
    data_path: str = "../../data/C8/cook"

    # 向量数据库配置
    collection_name: str = "RecipeChunk"
    qdrant_url: str = "http://localhost:6334"

    # mongodb配置
    mongodb_uri: str = os.getenv('MONGODB_URI', '')
    mongodb_database: str = "dish_agent"
    documents_collection: str = "recipe"
    chunks_collection: str = "recipe_chunks"
    
    # 模型配置
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    llm_model: str | None = os.getenv('MOONSHOT_MODEL_ID')
    hyde_llm_config: dict = field(  # @dataclass 不支持嵌套的默认值（因为会共享空间，所以使用field和default_factory来提供默认值
        default_factory=lambda: {
        'model_name': os.getenv('MOONSHOT_MODEL_FAST'), 
        'temperature': 0.4,
    })
    
    reranker_config: dict = field(
        default_factory=lambda: {
            'model_name': 'BAAI/bge-reranker-v2-m3',
            'use_fp16': True, 
            'cache_dir': '/tmp/reranker-model/', 
        }
    )

    # 检索配置
    top_k: int = 5
    threshold_config: dict = field(
        default_factory=lambda: {
            'search_threshold': 0.5,  # 检索结果相关度阈值，低于这个值的文档块将被过滤掉
            'match_threshold': 0.4,   # 假设性菜谱匹配的相关度阈值，低于这个值的假设性菜谱将被过滤掉
        }
    )

    rerank_api_config: dict = field(
        default_factory=lambda: {
            'rerank_api_model': "qwen3-rerank",
            'search_rerank_instruct': "Given a food search query, retrieve relevant recipe passages that answer the query.",  # 用于rerank时查询与文档块相关性匹配的指令
            'match_rerank_instruct': "Retrieve semantically similar text.",   # 用于rerank时假设性菜谱相似度匹配的指令
        }
    )

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048
    stream_output_tags: list = field(default_factory=lambda: ['llm_stream_output'])

    def __post_init__(self):
        """初始化后的处理"""
        pass
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'data_path': self.data_path,
            'qdrant_url': self.qdrant_url,
            'collection_name': self.collection_name,
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'top_k': self.top_k,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }

# 默认配置实例
DEFAULT_CONFIG = RAGConfig()
