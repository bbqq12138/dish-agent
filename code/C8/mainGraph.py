from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Send

import json
import asyncio
import logging
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from rag_modules import GenerationIntegrationModule

from branchGraph import SubGraph

logger = logging.getLogger(__name__)

def clearable_reducer(exist_list: list, new_item: list | str) -> list:
    """自定义reducer函数，用于合并分支结果，并支持特殊的"清空"标志"""
    if isinstance(new_item, str) and new_item.lower() == "clear":  # 如果新项是特殊的"清空"标志，则返回一个空列表
        return []
    
    if isinstance(new_item, str):
        new_item = [new_item]  # 将单个字符串转换为列表
    return exist_list + new_item  # 如果新项是一个列表，则将其与现有列表合并


class MainState(MessagesState):
    query: str
    branch_queries: list[str]
    branch_categories: list[Literal["list", "detail", "general"]]
    branch_results: Annotated[list[str], clearable_reducer]  # 默认自动合并结果，若传入"clear"则清空之前的结果
    result: str


class MainGraph:
    def __init__(self, llm):
        self.app = None
        self.llm = llm
        self.compiled_subgraph = None


    async def generate_answer(self, state: MainState, config: RunnableConfig) -> dict:
        """生成最终回答"""
        if len(state['branch_queries']) <= 1:  # 如果只有一个查询，直接返回该查询的结果，无需再生成了
            return {"result": state['branch_results'][0] if state['branch_results'] else "抱歉，我没有找到相关信息。"}
        
        stream_output_tags = config.get('configurable', {}).get("stream_output_tags", [])
        result = await GenerationIntegrationModule.multi_query_summary(
            self.llm.with_config(tags=stream_output_tags),
            state["query"],
            state["branch_results"],
        )  # 将所有子查询的结果合并起来生成最终回答
        return {"result": result}
    

    class MultiQueryComposer(BaseModel):
        """用于将用户的查询分解为一个或多个子查询，方便后续检索"""
        analyze: str = Field(description="你对于用户查询的拆分思考过程")
        queries: list[str] = Field(description="每个元素分别是分解后的子查询")


    async def multi_query_composer(self, state: MainState) -> dict:
        """
        多查询分解器 - 将用户查询拆分成多个子查询

        Args:
            query: 用户查询

        Returns:
            子查询列表
        """
        multi_query_composer_prompt = f"""
你是一个美食系统的智能助手，美食系统负责回答用户关于菜品推荐、菜谱生成、菜品问答三方面的提问。

要求：
 - 用户的提问中可能包含多个问题，你需要对用户的问题进行分析并将其拆分成多个独立的子问题，以便后续针对每个子问题进行检索和回答
 - 你需要将拆分出来的子问题填入JSON对象中的 'queries' 字段，该字段类型为字符串列表

举例：
 - 用户提问：我想吃点清淡的菜，有什么推荐吗？还有宫保鸡丁怎么做？
   回答：["推荐清淡菜", "宫保鸡丁的做法"]
 - 用户提问：宫保鸡丁和红烧茄子哪个好吃？
   回答：["宫保鸡丁味道怎么样", "红烧茄子味道怎么样"]
 - 用户提问：宫保鸡丁味道如何？辣不辣？
   回答：["宫保鸡丁味道怎么样，宫保鸡丁辣不辣"]  # 这两个问题都针对宫保鸡丁，可以合并成一个子查询，方便后续检索时匹配到相关的菜品信息

注意：
 - 每个子问题应该是针对**一个菜品**，不要在一个子问题中包含多个菜品，否则后续检索时无法准确匹配到相关的菜品信息
 - 若用户问题是针对一个菜品的多个方面，可以将这些问题合并成一个子查询，一个字问题不要包含多个菜品"""

        messages = [
            {'role': 'system', 'content': multi_query_composer_prompt}, 
            {'role': 'user', 'content': state['query']}
        ]

        response = await self.llm.ainvoke(
            input = messages, 
            temperature=0.3,
            response_format = self.MultiQueryComposer,  # 底层chat.completion.parse会自动解析
        )

        # response.additional_kwargs['parsed'] 自动解析为 Pydantic 对象中的 queries 属性
        try:
            if hasattr(response, 'additional_kwargs') and response.additional_kwargs.get('parsed'):
                branch_queries = response.additional_kwargs['parsed'].queries
            else:
                branch_queries = json.loads(response.content).get('queries', [])
        except Exception as e:
            logger.debug(response.content)
            logger.debug("多查询分解出错了")
            raise Exception(e)

        branch_queries = branch_queries[:5]     # 最多拆分成5个子查询，避免过多分支

        return {'branch_queries': branch_queries}


    def parallel_retrieval_router(self, state: MainState) -> list[Send]:
        sends = []

        # 根据分类结果动态地并行发送检索请求
        for q, category in zip(state['branch_queries'], state['branch_categories']):
            if category in ["list", "detail", "general"]:
                sends.append(Send('generate_subquery', {
                    'subquery': q, 
                    'query_category': category, 
                    'num_sub_queries': len(state['branch_queries'])
                }))
        return sends


    async def multi_query_router(self, state: MainState) -> dict:
        """
        查询路由 - 根据查询类型选择不同的处理方式
        """
        query_router_prompt = f"""
根据用户的问题，将其分类为以下三种类型之一：

1. 'list'：用户想要获取菜品列表或推荐，只需要菜名
   例如：
    - 推荐几个素菜
    - 与茄子相关的菜有什么
    - 今天外面下雪了，吃什么菜好呢？
    - 我最近血糖有些高，有什么适合吃的菜吗？
                                                  
2. 'detail'：用户想要具体的制作方法或详细信息
   例如：
    - 宫保鸡丁怎么做                                          
    - 宫保鸡丁的制作步骤、需要什么食材
    - 宫保鸡丁的制作技巧
                                                  
3. 'general'：其他一般性问题
   例如：
    - 什么是川菜
    - 宫保鸡丁的卡路里多少
    - 做菜的基本技巧有哪些
    - 如何判断菜是否熟了

只返回分类结果：list、detail 或 general
若有多个查询，则以空格分隔，如: list detail list

用户问题: {'  '.join(state['branch_queries'])}

分类结果:"""
        class_res = await self.llm.ainvoke([SystemMessage(content=query_router_prompt)], temperature=0.1)
        try:
            branch_categories = list(class_res.content.strip().split(' '))
        except Exception as e:
            logger.debug(class_res)
            logger.debug("分类出现问题")
            raise Exception(e)

        logger.debug(state['branch_queries'])
        logger.debug(branch_categories)

        if len(branch_categories) != len(state['branch_queries']):
            raise Exception("分类结果数量与子查询数量不一致")

        return {'branch_categories': branch_categories}


    def compile_main_graph(self, data_module=None, index_module=None, retrieval_module=None, generation_module=None):
        memory = InMemorySaver()

        self.compiled_subgraph = SubGraph(self.llm, data_module, index_module, retrieval_module, generation_module)
        branch_graph = self.compiled_subgraph.compile_subgraph()  # 生成编译好的子图节点

        builder = StateGraph(MainState)
        builder.add_node('multi_query_composer', self.multi_query_composer)
        builder.add_node('multi_query_router', self.multi_query_router)
        builder.add_node('generate_subquery', branch_graph)
        builder.add_node('generate', self.generate_answer)

        builder.add_edge(START, 'multi_query_composer')
        builder.add_edge('multi_query_composer', 'multi_query_router')
        builder.add_conditional_edges(
            source='multi_query_router',
            path=self.parallel_retrieval_router,
            path_map=['generate_subquery']
        )
        builder.add_edge('generate_subquery', 'generate')
        builder.add_edge('generate', END)

        self.app = builder.compile(checkpointer=memory)

async def test():
    import os
    from dotenv import load_dotenv
    from langchain.chat_models import init_chat_model
    load_dotenv()

    llm = init_chat_model(
        model=os.getenv("MOONSHOT_MODEL_ID"),
        model_provider='openai',
        api_key=os.getenv("MOONSHOT_API_KEY"),
        base_url=os.getenv("MOONSHOT_BASE_URL"),
        temperature=0.2,
    )

    agent = MainGraph(llm)
    agent.compile_main_graph()

    while True:
        user_input = input("请输入你的问题（输入exit退出）：")
        if user_input.lower() == "exit":
            break

        if agent.app is None:
            logger.info("主图还没有创建成功")
            break

        response = await agent.app.ainvoke(
            input={"query": user_input}, # type: ignore
            config={"configurable": {"thread_id": "thread-1"}},   
        )
        logger.info("回答：", response['result'])


if __name__ == "__main__":
    asyncio.run(test())