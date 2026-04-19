"""
RAG系统主程序
"""

import os
import sys
import asyncio
import logging
import warnings
from pathlib import Path
from collections.abc import AsyncIterator
from typing import List, Literal, overload
from huggingface_hub import login
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig

from mainGraph import MainGraph

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

warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")


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
        self.main_graph: MainGraph | None = None
        self.llm = None

        # 检查数据路径
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        # 检查API密钥
        if not os.getenv("MOONSHOT_API_KEY"):
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")
    

    def _initialize_system(self):
        """初始化所有模块"""
        print("🚀  正在初始化RAG系统...")

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
        print("🤖  初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(llm=self.llm)


        # 4. 初始化主图运行配置
        self.main_graph_config = RunnableConfig(
            configurable={
                'thread_id': 'user0-conversation0', # 用于区分LangGraph中检查点，标识一个对话窗口
                'top_k': self.config.top_k, 
                'search_threshold': self.config.threshold_config['search_threshold'],
                'match_threshold': self.config.threshold_config['match_threshold'],
                'stream_output_tags': self.config.stream_output_tags,  # 用于标记哪些LLM输出需要流式传递给前端
            }
        )

        print("✅  系统初始化完成！")


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
            print("✅  成功加载已保存的向量索引！")
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
                self.config.rerank_api_config
            )

        # 7. 显示统计信息
        stats = self.data_module.get_statistics()
        print(f"\n📊  知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   菜品分类: {list(stats['categories'].keys())}")
        print(f"   难度分布: {stats['difficulties']}")

        print("✅  知识库构建完成！")



    async def ask_question(self, question: str, stream: bool = False) -> str | AsyncIterator[str]:
        """
        回答用户问题

        Args:
            question: 用户问题
            stream: 是否使用流式输出

        Returns:
            生成的回答或生成器
        """
        
        async def _stream_generator():
            """
            内部流式生成器
            
            要让一个函数既能返回字符串（str），又能返回生成器（AsyncIterator），必须把含有 yield 的流式逻辑剥离到一个单独的内部函数里。
            """

            if not self.main_graph or not self.main_graph.app:
                raise ValueError("请先初始化系统")

            header_sent = False
            
            async for event in self.main_graph.app.astream_events(
                input={'query': question, 'branch_results': "clear"}, 
                config=self.main_graph_config, 
                version="v2"
            ):
                for llm_with_config in self.config.stream_output_tags:
                    if llm_with_config in event.get('tags', []) and event['event'] == 'on_chat_model_stream':
                        if not header_sent:
                            header_sent = True
                            yield "🔍  查询完成！\n" + "\n📣  回答：\n"
                        yield event['data']['chunk'].content # type: ignore


        if stream == True:
            # 流式输出
            return _stream_generator() 
        else:
            # 普通输出
            result = await self.main_graph.app.ainvoke(input={'query': question, 'branch_results': "clear"}, config=self.main_graph_config)  # type: ignore
            return result.get('result', '')
    

    async def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print("🍽️  尝尝咸淡RAG系统 - 交互式问答  🍽️")
        print("=" * 60)
        print("💡  解决您的选择困难症，告别'今天吃什么'的世纪难题！")
        
        # 初始化系统
        self._initialize_system()
        
        # 构建知识库
        await self.build_knowledge_base()

        # 4. 编译LangGraph主图
        print("🔧  编译LangGraph主图...")
        self.main_graph = MainGraph(self.llm)
        self.main_graph.compile_main_graph(self.data_module, self.index_module, self.retrieval_module, self.generation_module)
        
        print("\n交互式问答 (输入'退出'结束):")
        
        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
                    break
                

                # 询问是否使用流式输出
                stream_choice = input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
                use_stream = stream_choice != 'n'

                print("\n🔍  正在查询中...请稍候...")
                if use_stream:
                    # 流式输出
                    stream_iter = await self.ask_question(user_input, stream=True)
                    if not isinstance(stream_iter, AsyncIterator):
                        print("错误: 预期得到一个异步生成器，但实际返回了一个非生成器对象。")
                        continue
                    async for output_chunk in stream_iter:
                        print(output_chunk, end="", flush=True)
                    
                    print("\n")
                else:
                    # 普通输出
                    answer = await self.ask_question(user_input, stream=False)
                    print("🔍  查询完成！\n")
                    print("📣  回答：")
                    print(f"{answer}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")
        
        print("\n👋  感谢使用尝尝咸淡RAG系统！")



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
