"""
RAG系统配置文件
"""

from dataclasses import dataclass
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

    # 模型配置
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    llm_model: str | None = os.getenv('MOONSHOT_MODEL_ID')
    hyde_llm_config = {
        'model_name': os.getenv('MOONSHOT_MODEL_FAST'), 
        'temperature': 0.4,
    }
    reranker_config = {
        'model_name': 'BAAI/bge-reranker-v2-m3',
        'use_fp16': True, 
        'cache_dir': '/tmp/reranker-model/', 
    }

    # 检索配置
    top_k: int = 5

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

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
