"""
RAG系统主程序
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from collections.abc import AsyncIterator
from typing import List, Literal, overload
from huggingface_hub import login
from langchain.chat_models import init_chat_model

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, RAGConfig
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule
)

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    format='%(message)s',
)
logger = logging.getLogger(__name__)

# 降低第三方库的日志级别，只显示警告和错误
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("jieba").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)


# 登录Hugging Face Hub（如果需要）
if "HF_TOKEN" not in os.environ:
    login(token=os.getenv("HF_TOKEN"))

class RecipeRAGSystem:
    """食谱RAG系统主类"""

    def __init__(self, config: RAGConfig | None = None):
        """
        初始化RAG系统

        Args:
            config: RAG系统配置，默认使用DEFAULT_CONFIG
        """
        self.config = config or DEFAULT_CONFIG
        self.data_module: DataPreparationModule | None = None
        self.index_module: IndexConstructionModule | None = None
        self.retrieval_module: RetrievalOptimizationModule | None = None
        self.generation_module: GenerationIntegrationModule | None = None
        self.llm = None

        # 检查数据路径
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        # 检查API密钥
        if not os.getenv("MOONSHOT_API_KEY"):
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")
    
    def initialize_system(self):
        """初始化所有模块"""
        print("🚀 正在初始化RAG系统...")

         # 0.初始化llm
        print("初始化llm...")
        self._setup_llm()

        # 1. 初始化数据准备模块
        print("初始化数据准备模块...")
        self.data_module = DataPreparationModule(self.config.data_path)

        # 2. 初始化索引构建模块
        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            dense_model_name=self.config.embedding_model,
            collection_name=self.config.collection_name,
            qdrant_url=self.config.qdrant_url
        )

        # 3. 初始化生成集成模块
        print("🤖 初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(llm=self.llm)

        print("✅ 系统初始化完成！")


    def _setup_llm(self):
        """初始化大语言模型"""
        logger.info(f"正在初始化LLM: {self.config.llm_model}")

        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")

        self.llm = init_chat_model(
            model=self.config.llm_model or os.getenv("MOONSHOT_MODEL_ID"),
            model_provider='openai',
            api_key=os.getenv("MOONSHOT_API_KEY"),
            base_url=os.getenv("MOONSHOT_BASE_URL"),
            temperature=self.config.temperature,
        )

    
    async def build_knowledge_base(self):
        """构建知识库"""
        print("\n正在构建知识库...")
        if self.index_module is None or self.data_module is None:
            raise ValueError("请先初始化系统")

        # 1. 尝试加载已保存的索引
        if await self.index_module.load_index():
            print("✅ 成功加载已保存的向量索引！")
            # 仍需要加载文档以获取统计信息和后续使用
            print("加载食谱文档并进行分块...")
            self.data_module.load_documents()
            self.data_module.chunk_documents()
        else:
            print("未找到已保存的索引，开始构建新索引...")

            # 2. 加载文档
            print("加载食谱文档...")
            self.data_module.load_documents()

            # 3. 文本分块
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()

            # 4. 构建并保存向量索引
            print("构建并保存向量索引...")
            await self.index_module.build_vector_index(chunks)

        # 6. 初始化检索优化模块
        print("初始化检索优化...")
        if self.index_module.qdrant_client is not None:
            self.retrieval_module = RetrievalOptimizationModule(
                self.data_module, 
                self.index_module.qdrant_client, 
                self.config.collection_name, 
                self.llm, 
                self.config.hyde_llm_config,
                self.config.reranker_config
            )

        # 7. 显示统计信息
        stats = self.data_module.get_statistics()
        print(f"\n📊 知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   菜品分类: {list(stats['categories'].keys())}")
        print(f"   难度分布: {stats['difficulties']}")

        print("✅ 知识库构建完成！")


    
    @overload
    async def ask_question(self, question: str, stream: Literal[False] = False) -> str:  # type: ignore[overload-overlap]
        ...

    @overload
    async def ask_question(self, question: str, stream: Literal[True]) -> AsyncIterator[str]:
        ...

    async def ask_question(self, question: str, stream: bool = False) -> str | AsyncIterator[str]:
        """
        回答用户问题

        Args:
            question: 用户问题
            stream: 是否使用流式输出

        Returns:
            生成的回答或生成器
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")
        
        print(f"\n❓ 用户问题: {question}")

        if self.generation_module is None or self.retrieval_module is None or self.data_module is None:
            raise ValueError("模块未初始化")

        generation_module = self.generation_module
        retrieval_module = self.retrieval_module
        data_module = self.data_module

        # 1. 查询路由
        route_type = self.generation_module.query_router(question)
        print(f"🎯 查询类型: {route_type}")

        # 2. 智能查询重写（根据路由类型）
        if route_type == 'list':
            # 列表查询保持原查询
            rewritten_query = question
            print(f"📝 列表查询保持原样: {question}")
        else:
            # 详细查询和一般查询使用智能重写
            print("🤖 智能分析查询...")
            rewritten_query = self.generation_module.query_rewrite(question)
        
        # 3. 文档向量检索，并重排结果
        print("🔍 检索相关文档...")
        if route_type == 'list':
            relevant_chunks = await retrieval_module.hyde_search(rewritten_query, top_k=self.config.top_k)  # 三个假设性文档嵌入 + 混合搜索 + reranker重排
        else:
            relevant_chunkses = await retrieval_module.hybrid_search(rewritten_query, top_k=self.config.top_k)  # 全检索，自动结合元数据过滤和混合检索，重排使用的RRF算法，也可以选择reranker
            unrerank_relevant_chunks = [doc for sublist in relevant_chunkses for doc in sublist]  # 展平列表
            relevant_chunks = await retrieval_module.rerank_model_rerank(rewritten_query, unrerank_relevant_chunks, threshold=0.5)  # 对chunks进行reranker重排，过滤不相关文档块


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

            print(f"\n找到 {len(relevant_chunks)} 个相关文档块: {', '.join(doc_info)}")


        # 4. 检索父文档并去重
        doc_ids = []
        for chunk in relevant_chunks:
            parent_id = chunk.metadata.get('parent_id')
            if parent_id is not None and isinstance(parent_id, str):
                doc_ids.append(parent_id)
        relevant_docs = data_module.get_documents(doc_ids)  # 去重+根据父文档ID获取完整文档信息


        # 5. 检查是否找到相关内容并显示找到的文档名称
        if not relevant_docs:
            print("抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。")
        else:
            doc_names = []
            for doc in relevant_docs:
                dish_name = doc.metadata.get('dish_name', '未知菜品')
                doc_names.append(dish_name)
            if doc_names:
                print(f"找到文档: {', '.join(doc_names)}\n")

        # 5. 根据路由类型选择回答方式
        if stream:
            # 统一返回异步生成器，避免调用侧区分同步/异步生成器
            async def _stream_answer():
                if route_type == 'list':
                    print("📋 生成菜品列表...")
                    async for chunk in generation_module.generate_list_answer_stream(question, relevant_docs):
                        yield chunk
                    return

                print("✍️ 生成详细回答...")
                if route_type == "detail":
                    async for chunk in generation_module.generate_step_by_step_answer_stream(question, relevant_docs):
                        yield chunk
                else:
                    async for chunk in generation_module.generate_basic_answer_stream(question, relevant_docs):
                        yield chunk

            return _stream_answer() 

        if route_type == 'list':
            # 列表查询：直接返回菜品名称列表
            print("📋 生成菜品列表...")
            return await generation_module.generate_list_answer(question, relevant_docs)

        # 详细查询：获取完整文档并生成详细回答
        print("✍️ 生成详细回答...")

        if route_type == "detail":
            return await generation_module.generate_step_by_step_answer(question, relevant_docs)

        # 一般查询使用基础回答模式
        return await generation_module.generate_basic_answer(question, relevant_docs)
    

    def _extract_filters_from_query(self, query: str) -> dict:
        """
        从用户问题中提取元数据过滤条件
        """
        filters = {}
        # 分类关键词
        category_keywords = DataPreparationModule.get_supported_categories()
        for cat in category_keywords:
            if cat in query:
                filters['category'] = cat
                break

        # 难度关键词
        difficulty_keywords = DataPreparationModule.get_supported_difficulties()
        for diff in sorted(difficulty_keywords, key=len, reverse=True):
            if diff in query:
                filters['difficulty'] = diff
                break

        return filters
    
    def search_by_category(self, category: str, query: str = "") -> List[str]:
        """
        按分类搜索菜品
        
        Args:
            category: 菜品分类
            query: 可选的额外查询条件
            
        Returns:
            菜品名称列表
        """
        if not self.retrieval_module:
            raise ValueError("请先构建知识库")
        
        # 使用元数据过滤搜索
        search_query = query if query else category
        filters = {"category": category}
        
        docs = self.retrieval_module.metadata_filtered_search(search_query, filters, top_k=10) or []
        
        # 提取菜品名称
        dish_names = []
        for doc in docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            if dish_name not in dish_names:
                dish_names.append(dish_name)
        
        return dish_names
    
    # def get_ingredients_list(self, dish_name: str) -> str:
    #     """
    #     获取指定菜品的食材信息

    #     Args:
    #         dish_name: 菜品名称

    #     Returns:
    #         食材信息
    #     """
    #     if not all([self.retrieval_module, self.generation_module]):
    #         raise ValueError("请先构建知识库")

    #     # 搜索相关文档
    #     docs = self.retrieval_module.hybrid_search(dish_name, top_k=3)

    #     # 生成食材信息
    #     answer = self.generation_module.generate_basic_answer(f"{dish_name}需要什么食材？", docs)

    #     return answer
    
    async def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print("🍽️  尝尝咸淡RAG系统 - 交互式问答  🍽️")
        print("=" * 60)
        print("💡 解决您的选择困难症，告别'今天吃什么'的世纪难题！")
        
        # 初始化系统
        self.initialize_system()
        
        # 构建知识库
        await self.build_knowledge_base()
        
        print("\n交互式问答 (输入'退出'结束):")
        
        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
                    break
                

                # 询问是否使用流式输出
                stream_choice = input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
                use_stream = stream_choice != 'n'

                print("\n回答:")
                if use_stream:
                    # 流式输出
                    stream_iter = await self.ask_question(user_input, stream=True)
                    async for output_chunk in stream_iter:
                        print(output_chunk, end="", flush=True)
                    print("\n")
                else:
                    # 普通输出
                    answer = await self.ask_question(user_input, stream=False)
                    print(f"{answer}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")
        
        print("\n感谢使用尝尝咸淡RAG系统！")



async def main():
    """主函数"""
    try:
        # 创建RAG系统
        rag_system = RecipeRAGSystem()
        
        # 运行交互式问答
        await rag_system.run_interactive()
        
    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")

if __name__ == "__main__":
    asyncio.run(main())
