"""
检索优化模块
"""

import logging
import os
import json
import asyncio
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain.chat_models import init_chat_model, BaseChatModel
from FlagEmbedding import FlagReranker
from qdrant_client import models as qdrant_models
from .myQdrant import Qdrant
from .data_preparation import DataPreparationModule

logger = logging.getLogger(__name__)

class RetrievalOptimizationModule:
    """
    检索优化模块 - 包括假设性文档嵌入、混合检索、元数据过滤、Reranker重排四部分
    假设性文档嵌入(HyDE) - 让llm根据用户提问先生成3个假想的文档（这里是菜谱描述），然后用这3个文档去进行向量检索，最后重排
    混合检索 - 结合向量检索和BM25检索，使用RRF重排
    元数据过滤 - 让llm从用户查询或假设性文档中提取出元数据信息（比如菜系、口味、主食材等），然后在向量数据库中进行元数据过滤，最后再进行向量检索
    Reranker重排 - 使用Reranker模型对检索结果进行重排，以提高相关性

    PS：元数据过滤需要精心的设计和维护菜谱的元数据标签体系，才能发挥作用，否则可能会出现过度过滤或过滤不足的情况，导致检索结果不理想，这里目前还没有实现
    """
    
    def __init__(self, data_module: DataPreparationModule, qdrant_client: Qdrant, collection_name: str, llm, hyde_llm_config: dict, reranker_config: dict):
        """
        初始化检索优化模块
        
        Args:
            data_module: 数据模块
            qdrant_client: Qdrant向量数据库客户端
            collection_name: 向量数据库集合名称
            llm: 
            hyde_llm_config: 假设性文档生成模型配置
            reranker_config: 重排模型配置
        """
        self.data_module = data_module
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.llm: BaseChatModel = llm
        self.hyde_llm: BaseChatModel
        self._init_retrieval_module(hyde_llm_config, reranker_config)
    

    def _init_retrieval_module(self, hyde_llm_config:dict, reranker_config:dict):
        try: 
            self.hyde_llm = init_chat_model(
                model=hyde_llm_config['model_name'], 
                model_provider='openai',
                api_key=os.getenv("MOONSHOT_API_KEY"),
                base_url=os.getenv("MOONSHOT_BASE_URL"),
                temperature=hyde_llm_config['temperature'] 
            )
        except Exception as e:
            logger.error(f"HyDE LLM 初始化失败: {e}")
            raise ValueError("HyDE LLM 初始化失败")
        
        try: 
            self.qdrant_client.set_reranker(
                FlagReranker(
                    model_name_or_path=reranker_config.get('model_name', 'BAAI/bge-reranker-v2-m3'), 
                    use_fp16=reranker_config.get('use_fp16', True), 
                    cache_dir=reranker_config.get('cache_dir', '/tmp/reranker-model/'),
                    normalize=True, 
                )
            )
        except Exception as e:
            logger.error(f"Reranker 模型设置失败: {e}")
            raise ValueError("Reranker 模型设置失败")
        
        logger.info(f"HyDE LLM 初始化完成，使用模型: {hyde_llm_config['model_name']}，温度: {hyde_llm_config['temperature']}")
        logger.info(f"Reranker 模型设置完成，使用模型: {reranker_config.get('model_name', 'BAAI/bge-reranker-v2-m3')}, \
                    use_fp16: {reranker_config.get('use_fp16', True)}, cache_dir: {reranker_config.get('cache_dir', '/tmp/reranker-model/')}")
        
    
    async def hybrid_search(self, query: str | List[str], top_k: int = 5, vector_weight: float = 0.6) -> list[list[Document]]: # weight参数用于调整BM25和向量检索在RRF重排中的权重，默认BM25占0.4，向量检索占0.6
        """
        混合检索 - 结合向量检索和BM25检索，使用RRF重排

        Args:
            query: 查询文本
            top_k: 返回结果数量
            vector_weight: 向量检索的权重

        Returns:
            检索到的文档列表
        """
        if isinstance(query, str):
            query = [query]
        num_query = len(query)

        # 1. 生成query的稠密向量和稀疏向量
        if not self.qdrant_client.dense_model or not self.qdrant_client.sparse_model:
            logger.error("混合检索失败: 嵌入模型未设置")
            raise ValueError("混合检索失败: 嵌入模型未设置")
        query_dense_vecs, query_sparse_vecs = await asyncio.gather(
            self.qdrant_client.dense_model.aembed_documents(query),
            asyncio.to_thread(self.qdrant_client.sparse_model.embed_batch, query)  # jieba分词和稀疏向量生成是CPU密集型任务，使用to_thread放到线程池中执行，避免阻塞事件循环
        )

        # query_dense_vecs = self.qdrant_client.dense_model.embed_query(query)  # 同步方法
        # query_sparse_vecs = self.qdrant_client.sparse_model.embed_batch(query)        

        # 2. qdrant服务器进行混合检索，分别得到两路检索结果
        requests = []
        for query_dense_vec, query_sparse_vec in zip(query_dense_vecs, query_sparse_vecs):
            requests.append(
                qdrant_models.QueryRequest(
                    using="dense",
                    query=query_dense_vec,
                    limit=2 * top_k, 
                    with_payload=['parent_id']
                )
            )
            requests.append(
                qdrant_models.QueryRequest(
                    using='sparse', 
                    query=qdrant_models.SparseVector(
                        indices=query_sparse_vec.indices.tolist(),
                        values=query_sparse_vec.values.tolist()
                    ), 
                    limit=2 * top_k, 
                    with_payload=['chunk_id', 'parent_id']
                )
            )

        results = await self.qdrant_client.query_batch_points(
            collection_name=self.collection_name, 
            requests=requests, 
        )

        # 3. 从两路检索结果中提取出父文档ID，并获取对应的父文档内容，并进行RRF重排
        rrf_reranked_docs = []
        for i in range(num_query):
            vector_ids = list(point.payload['chunk_id'] for point in results[i*2].points if point.payload.get('chunk_id'))  # type: ignore # 未去重
            bm25_ids = list(point.payload['chunk_id'] for point in results[i*2+1].points if point.payload.get('chunk_id')) # type: ignore
        
            vector_docs = self.data_module.get_chunks(vector_ids) # 去重？
            bm25_docs = self.data_module.get_chunks(bm25_ids)

            rrf_reranked_docs.append(self._rrf_rerank(vector_docs, bm25_docs, top_k=top_k, vector_weight=vector_weight))  # 每个查询都进行RRF重排，并选取top_k个结果
    

        return rrf_reranked_docs # 返回的是chunks的列表
        
    

    async def rerank_model_rerank(self, query: str, candidate_docs: List[Document], top_k: int=5, threshold: float=0.6) -> List[Document]:
        """
        对候选文档进行重排，这里使用 cross-encoder reranker 对候选文档进行打分，并根据分数进行排序，选择 top_k 个文档返回
        """
        text_pairs = []
        for doc in candidate_docs:
            text_pairs.append((query, doc.page_content))
        
        if not self.qdrant_client.reranker:
            logger.error("Reranker模型未设置，无法进行重排")
            raise ValueError("Reranker模型未设置，无法进行重排")
        
        scores = await asyncio.to_thread(self.qdrant_client.reranker.compute_score, text_pairs)
        if scores is None:
            logger.error("重排过程中出现错误，无法获取文档得分")
            raise ValueError("重排过程中出现错误，无法获取文档得分")
        scores = [(score, i) for i, score in enumerate(scores)]
        scores.sort(reverse=True, key=(lambda x: x[0]))
        logger.info(scores)

        reranked_docs = []
        count = 0
        for score, seq in scores:
            if score < threshold:     # 这里设置一个排序模型的分数阈值，只有当文档与查询的相关性得分超过threshold时才被认为是相关的，才会被加入到重排结果中
                continue
            reranked_docs.append(candidate_docs[seq])
            count += 1
            if count >= top_k:
                break

        logger.info(f"重排完成: 输入候选文档{len(candidate_docs)}个, 输出重排后文档{len(reranked_docs)}个")
        return reranked_docs
    

    # def all_retrieval(self, query: str, weight=[0.4, 0.6]) -> List[Document]:
    #     """
    #     全检索 - 先进行元数据过滤，再在其上层进行BM25和稠密向量检索，最后混合排序
    #     这里为了简单，分 元数据过滤+稠密向量检索 和 混合检索 两路，最终重排序（当然这并不完全合理）
    #     """
    #     metadata_filtered_docs = self.metadata_filtered_search_(query)
    #     hybrid_docs = self.hybrid_search(query, top_k=5, weight=weight)  # 先获取更多的候选文档，后续再重排过滤

    #     revelant_docs = self._unique_documents(metadata_filtered_docs, hybrid_docs)    # 合并两个检索器的结果
        
    #     logger.info(f"基于元数据的自检索器，找到 {len(metadata_filtered_docs)} 个相关文档块, 基于混合检索器，找到 {len(hybrid_docs)} 个相关文档块, 合并后共有 {len(revelant_docs)} 个相关文档块")
        
    #     return revelant_docs

    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], top_k: int = 5, k: int = 60, vector_weight: float = 0.6) -> List[Document]:
        """
        使用RRF (Reciprocal Rank Fusion) 算法重排文档

        Args:
            vector_docs: 向量检索结果
            bm25_docs: BM25检索结果
            k: RRF参数，用于平滑排名
            vector_weight: 向量检索的权重

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
            rrf_score = vector_weight * 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"向量检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 计算BM25检索结果的RRF分数
        bm25_weight = 1 - vector_weight
        for rank, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc

            rrf_score = bm25_weight * 1.0 / (k + rank + 1)
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
        reranked_docs = reranked_docs[:top_k]  # 选取top_k个结果

        logger.debug(f"RRF重排完成: 向量检索{len(vector_docs)}个文档, BM25检索{len(bm25_docs)}个文档, 合并后{len(reranked_docs)}个文档")

        return reranked_docs

    def _unique_documents(self, docs1: List[Document], docs2: List[Document]) -> List[Document]:
        """合并两个文档列表，并去重，去重的依据是文档的父文档ID（parent_id），如果两个文档的parent_id相同，则认为是重复的，只保留一个"""
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

    async def hyde_search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        HYDE检索 - 先让大模型根据查询生成一个假想的文档，然后用这个文档去进行向量检索，最后重排

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            检索到的文档列表
        """

        # 1. 生成假想文档
        hyde_cot_prompt = f"""
你是一个顶级的 AI 饮食推荐引擎。你的任务是深度解析用户的饮食需求，进行意图推理后，生成**三个**相关的、极具画面感的菜品描述（假设性菜谱文档），用于后续的向量数据库检索。

【生成要求】
1. 必须输出且仅输出一个合法的 JSON 对象，不要包含任何多余的解释文字或 Markdown 标记符号（如 ```json）。
2. JSON 对象必须包含以下两个字段：
   - "analyze": (字符串) 你的思考过程。简要分析用户的真实意图、情绪场景、所需的营养搭配，以及推导应该采用哪三种不同的烹饪手法/食材组合来满足用户。
   - "hyde_documents": (JSON数组, 数组内是3个字符串) 包含 3 个不同的菜谱描述。每个描述 50-100 字。
3. ⚠️ 对于 hyde_documents 的极其重要要求：描述中要大量使用【烹饪动词】（如：大火爆香、慢炖、收汁）、【感官形容词】（如：外酥里嫩、汤汁浓郁、清脆解腻）和【具体食材类别】。模仿真实菜谱的正文口吻，绝对不要写干瘪的标签。

【Few-Shot 示例】

输入：下班好累，想做个有肉的快手菜，最好十分钟搞定。
输出：
{{
    "analyze": "用户当前状态是疲惫，核心诉求是‘有肉’和‘极度快手(10分钟内)’。为了满足需求，我需要避开耗时的炖煮，选择三种最高效的烹饪策略：1. 猛火爆炒薄肉片；2. 微波炉或蒸锅免洗锅做法；3. 平底锅高温干煎肉排。",
    "hyde_documents": [
        "热锅冷油，将腌制好的肉片大火快速滑炒至变色。加入葱姜蒜爆香，淋入生抽和少许老抽上色，翻炒均匀。整个过程只需几分钟，肉质鲜嫩多汁，浓郁的酱汁极其下饭，是一道极其抚慰人心的快手爆炒肉菜。",
        "不需要复杂的起锅烧油，将切成薄片的肉类和洗净的快熟蔬菜码放在深盘中，淋上调配好的葱油或蒜蓉酱汁。直接放入微波炉高温加热几分钟，或者上蒸锅大火速蒸。肉质软嫩入味，汤汁拌饭绝佳，做法极度省事且毫无油烟。",
        "平底锅大火烧热，直接将带少许脂肪的肉块或肉排下锅，煎至两面金黄微焦，逼出多余油脂。只撒上简单的海盐和现磨黑胡椒提味。外皮酥脆，内里鲜嫩爆汁，做法粗犷快手，大口吃肉极大缓解了一天的疲惫。"
    ]
}}

输入：最近在减脂期，想吃点清淡低卡的，但要有饱腹感。
输出：
{{
    "analyze": "核心诉求是‘减脂低卡’和‘高饱腹感’。低卡意味着烹饪要少油无油（如凉拌、清汤、蒸煮），高饱腹感则需要优质蛋白（白肉、豆制品）和高纤维（粗粮、菌菇）。我将提供凉拌高蛋白、清透鲜蔬汤和粗粮蒸煮三种方向的描述。",
    "hyde_documents": [
        "这是一道清爽解腻的低卡凉拌菜。将富含优质高蛋白的低脂白肉水煮断生后撕成细丝，搭配大量清脆的高纤维蔬菜（如黄瓜、木耳）。淋上由柠檬汁、生抽和少许代糖调制的无油醋汁。口感酸甜开胃，咀嚼感极强，吃一大盘也毫无罪恶感。",
        "一道热气腾腾的低脂鲜蔬汤。砂锅中加入清水，放入滑嫩的豆腐、菌菇和各种时令绿叶菜一起大火煮沸。全程不滴一滴明油，只用少许盐和白胡椒粉简单调味。汤汁清透鲜美，菌菇自带氨基酸提鲜。热乎乎地下肚，既暖胃又能提供极强的饱腹感。",
        "将富含优质碳水的根茎类粗粮（如红薯、南瓜）与低脂高蛋白食材一起上蒸锅清蒸。这种烹饪方式保留了食物最原始的清甜与本味。出锅后只需蘸取少许清淡的蘸水食用。少油少盐，营养比例完美，饱腹感持久，是非常优秀的减脂期正餐替代品。"
    ]
}}

输入：{query}
输出："""

        hyde_result = await self.hyde_llm.ainvoke([SystemMessage(content=hyde_cot_prompt)])
        
        # 2. 生成的结果为JSON对象，解析出hyde_recipes字段，作为后续检索的查询条件
        try:
            hyde_recipes = json.loads(s=hyde_result.content) # type: ignore
        except json.JSONDecodeError as e:
            logger.error(f"HyDE生成的文档不是合法的JSON格式: {e}")
            raise ValueError("HyDE生成的文档不是合法的JSON格式")

        analyze = hyde_recipes.get('analyze', '')
        hyde_docs = hyde_recipes.get('hyde_documents', [])
        logger.info(f"HyDE分析结果: {analyze}\nHyDE生成了 {len(hyde_docs)} 个假想文档")
        logger.debug(f"HyDE生成的假想文档: {hyde_docs}")

        # 3. 用生成的hyde_docs去进行向量检索，获取相关文档
        reranked_docs = []
        if hyde_docs:
            res_docss = await self.hybrid_search(hyde_docs, top_k=2 * top_k  // len(hyde_docs))   # 这里直接用混合检索，利用生成的hyde_docs去进行向量检索，获取相关文档
            for i, docs in enumerate(res_docss):
                logger.info(f"使用假想文档 {i+1} 进行检索，得到 {len(docs)} 个相关文档")
                reranked_docs.append(await self.rerank_model_rerank(hyde_docs[i], docs, top_k=top_k, threshold=0.3))
            # res_docs = [doc for docs in res_docss for doc in docs]  # 展平列表
            # logger.info(f"检索到 {len(res_docs)} 个相关文档"  )
        else:
            res_docss = await self.hybrid_search(query, top_k=top_k, vector_weight=0.7)   # 如果HyDE没有生成有效的文档，就退化到普通的混合检索
            res_docs = res_docss[0]
        
        # 4. 重排并返回结果
        
        logger.debug(f"🔍HyDE检索完成:  检索并重排后返回{len(reranked_docs)}个文档")
        return reranked_docs

    def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 5):
        pass
    # def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
    #     """
    #     利用自检索器/LLM从query中提取元数据信息, 进行元数据过滤
    #     元数据过滤通常是由向量数据库完成，用户查询时传入filter，令向量数据库实际执行元数据过滤后再进行向量检索
    #     比较复杂的是如何给菜谱设计一个合理的元数据标签体系，不合理的标签可能会限制检索结果，导致过滤不足或过渡过滤
        
        
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

