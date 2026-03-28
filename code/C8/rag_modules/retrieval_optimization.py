"""
检索优化模块
"""

import logging
from typing import List, Dict, Any
import jieba
import os

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever

from langchain_community.cross_encoders.huggingface import HuggingFaceCrossEncoder


logger = logging.getLogger(__name__)

class RetrievalOptimizationModule:
    """
    检索优化模块 - 包括混合检索和元数据过滤两部分
    正常流程应该是先进行元数据过滤，再在其上层进行BM25和稠密向量检索，最后混合排序
    这在Milvus中可以自动完成，但是对于BM25检索器和Chroma却要手动组织
    这里为了简单，分 元数据过滤+稠密向量检索 和 混合检索 两路，最终重排序（当然这并不完全合理）
    """
    
    
    def __init__(self, vectorstore: Chroma, chunks: List[Document], llm):
        """
        初始化检索优化模块
        
        Args:
            vectorstore: Chroma向量存储
            chunks: 文档块列表
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.llm = llm
        self.reranker_model = None
        self.reranker = None
        self.setup_retrievers()

    def setup_retrievers(self):
        """设置向量检索器和BM25检索器"""
        logger.info("正在设置检索器...")

        # 1.初始化向量检索器
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # 2.初始化 BM25 检索器，并强制使用中文分词
        self.bm25_retriever = BM25Retriever.from_documents(
            documents=self.chunks,
            preprocess_func=self._jieba_tokenizer,  # 关键就在这一行！
            k=5
        )

        # 定义元数据字段信息，供自检索器使用
        metadata_field_info = [
            AttributeInfo(
                name='主标题', 
                description='这是菜谱中文本的主标题', 
                type='string'
            ), 
            AttributeInfo(
                name='二级标题', 
                description='这是菜谱中文本的二级标题', 
                type='string'
            ), 
            AttributeInfo(
                name='三级标题', 
                description='这是菜谱中文本的三级标题', 
                type='string'
            ), 
            AttributeInfo(
                name='category', 
                description='这是菜谱中菜的类别', 
                type='string'
            ), 
            AttributeInfo(
                name='dish_name', 
                description='这是菜谱中菜的名字', 
                type='string'
            ), 
            AttributeInfo(
                name='difficulty', 
                description='这是菜谱中该菜品的制作难度', 
                type='string'
            ), 
        ]

        # 3.初始化自检索器，允许大模型自己决定返回数量，并在控制台打印出模型生成的查询
        self.self_query_retriever = SelfQueryRetriever.from_llm(
            llm=self.llm, 
            vectorstore=self.vectorstore,
            document_contents="各种菜的菜谱", 
            metadata_field_info=metadata_field_info, 
            enable_limit=True,  # 允许大模型自己决定返回数量
            verbose=True,  # 在控制台打印出模型到底生成了什么查询
            search_kwargs={"k": 5}  # 控制检索器检索结果数量
        )

        # 4. 初始化重排模型
        self.reranker_model = HuggingFaceCrossEncoder(
            model_name='BAAI/bge-reranker-v2-m3', 
            model_kwargs={
                'device': 'cpu',
                'cache_folder': '/tmp/huggingface_cache',  # 指定缓存目录，避免默认位置磁盘空间不足
            }
        )
            
        logger.info("检索器设置完成")
    
    def hybrid_search(self, query: str, top_k: int = 3, weight=[0.4, 0.6]) -> List[Document]:
        """
        混合检索 - 结合向量检索和BM25检索，使用RRF重排

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            检索到的文档列表
        """
        # 分别获取向量检索和BM25检索结果
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        # 使用RRF重排
        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs, weight=weight)
        return reranked_docs[:top_k]

    def metadata_filtered_search_(self, query: str) -> List[Document]:
        """
        利用自检索器从query中提取元数据信息, 进行元数据过滤
        元数据过滤通常是由向量数据库完成，用户查询时传入filter，令向量数据库实际执行元数据过滤后再进行向量检索，当然元数据
        """
        return self.self_query_retriever.invoke(query)

    def rerank(self, query: str, candidate_docs: List[Document], top_k: int=5, threshold: float=0.6) -> List[Document]:
        """
        对候选文档进行重排，这里简单使用LLM对候选文档进行打分，并根据分数进行排序，选择前5个文档返回
        由于原本的documents的page_content不包含标题等元数据，我们可以在这里把文档内容和元数据拼接起来，形成新的文本输入给重排模型
        若不在文本中载入元数据，reranker得出的结果可能不太准确
        PS：这里先不弄了，以后再弄
        """
        text_pairs = []
        for doc in candidate_docs:
            text_pairs.append((query, doc.page_content))
        scores = self.reranker_model.score(text_pairs)
        scores = [(score, i) for i, score in enumerate(scores)]
        scores.sort(reverse=True, key=(lambda x: x[0]))

        reranked_docs = []
        count = 0
        for score, seq in scores:
            if score < threshold:     # 这里设置一个排序模型的分数阈值，只有当文档与查询的相关性得分超过threshold时才被认为是相关的，才会被加入到重排结果中
                continue
            reranked_docs.append(candidate_docs[seq])
            count += 1
            if count >= top_k:
                break

        # logger.info(f"重排完成: 输入候选文档{len(candidate_docs)}个, 输出重排后文档{len(reranked_docs)}个")
        return reranked_docs

    def all_retrieval(self, query: str, weight=[0.4, 0.6]) -> List[Document]:
        """
        全检索 - 先进行元数据过滤，再在其上层进行BM25和稠密向量检索，最后混合排序
        这里为了简单，分 元数据过滤+稠密向量检索 和 混合检索 两路，最终重排序（当然这并不完全合理）
        """
        metadata_filtered_docs = self.metadata_filtered_search_(query)
        hybrid_docs = self.hybrid_search(query, top_k=5, weight=weight)  # 先获取更多的候选文档，后续再重排过滤

        revelant_docs = self._unique_documents(metadata_filtered_docs, hybrid_docs)    # 合并两个检索器的结果
        
        logger.info(f"基于元数据的自检索器，找到 {len(metadata_filtered_docs)} 个相关文档块, 基于混合检索器，找到 {len(hybrid_docs)} 个相关文档块, 合并后共有 {len(revelant_docs)} 个相关文档块")
        
        return revelant_docs

    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], k: int = 60, weight=[0.4, 0.6]) -> List[Document]:
        """
        使用RRF (Reciprocal Rank Fusion) 算法重排文档

        Args:
            vector_docs: 向量检索结果
            bm25_docs: BM25检索结果
            k: RRF参数，用于平滑排名

        Returns:
            重排后的文档列表
        """
        doc_scores = {}
        doc_objects = {}

        # 计算向量检索结果的RRF分数
        for rank, doc in enumerate(vector_docs):
            # 使用文档内容的哈希作为唯一标识
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc

            # RRF公式: 1 / (k + rank)
            rrf_score = weight[1] * 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"向量检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 计算BM25检索结果的RRF分数
        for rank, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc

            rrf_score = weight[0] * 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"BM25检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 按最终RRF分数排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # 构建最终结果
        reranked_docs = []
        for doc_id, final_score in sorted_docs:
            if doc_id in doc_objects:
                doc = doc_objects[doc_id]
                # 将RRF分数添加到文档元数据中
                doc.metadata['rrf_score'] = final_score
                reranked_docs.append(doc)
                logger.debug(f"最终排序 - 文档: {doc.page_content[:50]}... 最终RRF分数: {final_score:.4f}")

        logger.info(f"RRF重排完成: 向量检索{len(vector_docs)}个文档, BM25检索{len(bm25_docs)}个文档, 合并后{len(reranked_docs)}个文档")

        return reranked_docs

    def _jieba_tokenizer(self, text: str) -> list[str]:
        """定义中文分词函数(BM25方法针对的是英文)"""
        # jieba.lcut 会把 "番茄炒蛋怎么做" 切分成 ['番茄', '炒蛋', '怎么', '做']
        return jieba.lcut(text)

    def _unique_documents(self, docs1: List[Document], docs2: List[Document]) -> List[Document]:
        """由于BM25和向量检索的结果可能有重复（尤其是当文档数量不大时），我们需要一个函数来合并两个结果列表，并去除重复的文档"""
        unique_docs = []
        for doc2 in docs2:
            is_repeat = False
            for doc1 in docs1:
                if doc2.metadata.get('chunk_id') == doc1.metadata.get('chunk_id'):
                    is_repeat = True
                    break
            
            if not is_repeat:
                unique_docs.append(doc2)

        return unique_docs + docs1


    # def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
    #     """
    #     带元数据过滤的检索
        
    #     Args:
    #         query: 查询文本
    #         filters: 元数据过滤条件
    #         top_k: 返回结果数量
            
    #     Returns:
    #         过滤后的文档列表
    #     """
    #     # 先进行混合检索，获取更多候选
    #     docs = self.hybrid_search(query, top_k * 3)
        
    #     # 应用元数据过滤
    #     filtered_docs = []
    #     for doc in docs:
    #         match = True
    #         for key, value in filters.items():
    #             if key in doc.metadata:
    #                 if isinstance(value, list):
    #                     if doc.metadata[key] not in value:
    #                         match = False
    #                         break
    #                 else:
    #                     if doc.metadata[key] != value:
    #                         match = False
    #                         break
    #             else:
    #                 match = False
    #                 break
            
    #         if match:
    #             filtered_docs.append(doc)
    #             if len(filtered_docs) >= top_k:
    #                 break
        
    #     return filtered_docs

