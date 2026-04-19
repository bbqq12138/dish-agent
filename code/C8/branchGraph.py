from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import tool_node

from langchain_core.documents import Document
from langchain_core.runnables.config import RunnableConfig
from typing import Literal
import logging

from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule
)

logger = logging.getLogger(__name__)


class BranchState(MessagesState):
        subquery: str
        num_sub_queries: int
        query_category: Literal["list", "detail", "general"]
        relevant_chunks: list[Document]
        relevant_docs: list[Document]
        branch_results: list[str]  # 子图没有这个字段，则会自动过滤掉这个字段，所以不会传入主图。因此在子图最后一节点中附加这个字段并返回


class SubGraph:
    def __init__(self, llm, 
                 data_module: DataPreparationModule | None = None, 
                 index_module: IndexConstructionModule | None = None, 
                 retrieval_module: RetrievalOptimizationModule | None = None, 
                 generation_module: GenerationIntegrationModule | None = None, 
                 llm_other=None
    ):
        self.llm = llm  # 并发的子图节点异步共享同一llm，但是可能同一API厂商会有并发限制
        self.data_module = data_module
        self.index_module = index_module
        self.retrieval_module = retrieval_module
        self.generation_module = generation_module
        self.llm_other = llm_other  # 备用API厂商


    def compile_subgraph(self) -> CompiledStateGraph:
        """供主图调用生成子图节点"""
        branch_builder = StateGraph(BranchState)

        branch_builder.add_node('retrieve_chunks', self.retrieve_chunks)
        branch_builder.add_node('fetch_parent_docs', self.fetch_parent_docs)
        branch_builder.add_node('generate_sub_answer', self.generate_sub_answer)

        branch_builder.add_edge(START, 'retrieve_chunks')
        branch_builder.add_edge('retrieve_chunks', 'fetch_parent_docs')
        branch_builder.add_edge('fetch_parent_docs', 'generate_sub_answer')
        branch_builder.add_edge('generate_sub_answer', END)

        return branch_builder.compile()


    async def query_rewriter(self, state: BranchState) -> dict:
        """查询重写节点 - 让大模型判断是否需要重写查询，并返回重写后的查询"""
        if not self.generation_module:
            raise Exception("GenerationIntegrationModule is required for query rewriting")
        
        # 0. 多查询分解（主图）
        # 1. 查询路由（主图）
        # 2. 智能查询重写（根据路由类型）
        if state['query_category'] == 'list':
            # 列表查询保持原查询
            rewritten_query = state['subquery']
            logger.debug(f"📝 列表查询保持原样: {state['subquery']}")
        else:
            # 详细查询和一般查询使用智能重写
            logger.debug("🤖 智能分析查询...")
            rewritten_query = await self.generation_module.query_rewrite(state['subquery'])

        return {'subquery': rewritten_query}
    

    async def retrieve_chunks(self, state: BranchState, config: RunnableConfig) -> dict:
        """文档检索节点 - 根据查询进行文档检索，并返回相关文档子块"""
        if not self.retrieval_module:
            raise Exception("RetrievalOptimizationModule is required for document retrieval")
        
        top_k = config.get('configurable', {}).get('top_k', 5)  # 从配置中获取top_k参数，默认为5
        instruct = config.get('configurable', {}).get('search_rerank_instruct', '')  # 从配置中获取rerank指令，默认为空字符串
        search_threshold = config.get('configurable', {}).get('search_threshold', 0.5)  # 从配置中获取检索相关度阈值，默认为0.5
        match_threshold = config.get('configurable', {}).get('match_threshold', 0.4)  # 从配置中获取假设性菜谱匹配相关度阈值，默认为0.4
        
         # 3. 文档向量检索，并重排结果
        logger.debug(f"\n[分支{state['subquery']}] 🔍 正在检索...")
        if state['query_category'] == 'list':
            relevant_chunks = await self.retrieval_module.hyde_search(state['subquery'], top_k, search_threshold, match_threshold)  # 三个假设性文档嵌入 + 混合搜索 + reranker重排
        else:
            relevant_chunkses = await self.retrieval_module.hybrid_search(state['subquery'], top_k=top_k)  # 全检索，自动结合元数据过滤和混合检索，重排使用的RRF算法，也可以选择reranker
            unrerank_relevant_chunks = [doc for sublist in relevant_chunkses for doc in sublist]  # 展平列表

            relevant_chunks = await self.retrieval_module.api_rerank(
                query=state['subquery'], 
                candidate_docs=unrerank_relevant_chunks, 
                instruct=instruct,
                threshold=search_threshold
            )  # 对chunks进行reranker重排，过滤不相关文档块


        # 显示检索到的子块信息
        if relevant_chunks:
            doc_info = []
            for doc in relevant_chunks:
                dish_name = doc.metadata.get('dish_name', '未知菜品')
                title = []
                for t in ['主标题', '二级标题', '三级标题']:
                    if doc.metadata.get(t):
                        title.append(doc.metadata.get(t))
                doc_info.append(f"{dish_name}({'-'.join(title)})")

            logger.debug(f"[分支{state['subquery']}] 🎯 检索到 {len(relevant_chunks)} 个相关文档块: {', '.join(doc_info)}")

        return {'relevant_chunks': relevant_chunks}


    async def fetch_parent_docs(self, state: BranchState) -> dict:
        """父文档获取节点 - 根据相关文档块获取对应的父文档"""
        relevant_chunks = state['relevant_chunks']

        if not relevant_chunks:
            logger.debug(f"[分支{state['subquery']}] ⚠️ 未找到任何子块，跳过父文档检索。")
            return {"relevant_docs": []}

        if not self.data_module:
            raise Exception("DataPreparationModule is required for fetching parent documents")

        # 4. 检索父文档并去重
        doc_ids = []
        for chunk in relevant_chunks:
            parent_id = chunk.metadata.get('parent_id')
            if parent_id is not None and isinstance(parent_id, str):
                doc_ids.append(parent_id)
        relevant_docs = await self.data_module.get_documents(doc_ids)  # 去重+根据父文档ID获取完整文档信息

        # 5. 检查是否找到相关内容并显示找到的文档名称
        if relevant_docs:
            doc_names = [doc.metadata.get('dish_name', '未知') for doc in relevant_docs]
            logger.debug(f"[分支{state['subquery']}] 🎯 成功映射完整父文档: {', '.join(doc_names)}")
        else:
            logger.debug(f"[分支{state['subquery']}] ❌ 未能找到对应的父文档。")
        
        return {'relevant_docs': relevant_docs}


    async def generate_sub_answer(self, state: BranchState, config: RunnableConfig) -> dict:
        """答案生成节点 - 根据查询类型和相关文档生成回答"""
        question = state['subquery']
        relevant_docs = state.get('relevant_docs', [])
        route_type = state['query_category']
        stream_output_tags = config.get('configurable', {}).get("stream_output_tags", [])

        if not self.generation_module:
            raise Exception("GenerationIntegrationModule is required for answer generation")
        
        if not relevant_docs:
            return {"branch_results": [f"关于“{question}”，抱歉没有找到相关的食谱信息。"]}

        if state['num_sub_queries'] <= 1:
            tagged_llm = self.llm.with_config(tags=stream_output_tags)
        else: tagged_llm = self.llm

        # 6. 根据路由类型选择回答方式
        logger.debug(f"[分支{state['subquery']}] ✍️ 正在生成 {route_type} 类型回答...")
        if route_type == 'list':
            response = await GenerationIntegrationModule.generate_list_answer(
                tagged_llm, question, relevant_docs)
        elif route_type == 'detail':
            response = await GenerationIntegrationModule.generate_step_by_step_answer(
                tagged_llm, question, relevant_docs)
        else:
            response = await GenerationIntegrationModule.generate_basic_answer(
                tagged_llm, question, relevant_docs)

        return {"branch_results": response}


    async def test_branch_node(self, state: BranchState):
        """测试子图节点的调用"""
        result = f"子查询: {state['subquery']}, 查询类型: {state['query_category']}"
        logger.info(result)
        return {'branch_results': [result]}  # 注意这里必须返回这个字段，主图才会接收到这个字段并进行合并。子图要是没有这个字段，主图就会自动过滤掉这个字段，不会传入主图
