from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks.manager import dispatch_custom_event
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

def int_add_reducer(a: int | None, b: int | str) -> int:
    if a is None:
        a = 0
    
    if isinstance(b, str):
        if b.lower() == "clear":  # 如果新项是特殊的"清空"标志，则返回0
            return 0
        return a
    
    return a + b


class MainState(MessagesState):
    query: str
    branch_queries: list[str]
    branch_categories: list[Literal["list", "detail", "general"]]
    branch_results: Annotated[list[str], clearable_reducer]  # 默认自动合并结果，若传入"clear"则清空之前的结果
    relevant_docs_total: Annotated[int, int_add_reducer]  # 统计所有分支检索到的相关文档总数
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

        dispatch_custom_event(
            name="docs_num_info",
            data={"relevant_docs_total": state.get('relevant_docs_total', '未知')},
            config=config,  
        )
        
        stream_output_tags = config.get('configurable', {}).get("stream_output_tags", [])
        response = await GenerationIntegrationModule.multi_query_summary(
            self.llm.with_config(tags=stream_output_tags),
            state["query"],
            state["branch_results"],
        )  # 将所有子查询的结果合并起来生成最终回答

        return {"result": response.content, "messages": [response]}


    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


    async def query_rewrite_history(self, state: MainState) -> dict:
        """根据对话历史查询重写"""
        
        messages = state.get('messages', [])
        
        # 💡 拦截器：如果没有历史对话（说明是第一轮提问），直接跳过重写，节省 Token 和时间！
        # 假设 messages 里只包含了之前的历史，而当前最新的 query 在 state['query'] 中
        # 或者如果 messages 里包含了最新的提问，那判断条件就是 len(messages) <= 1
        if not messages:
            return {'messages': [HumanMessage(content=state['query'])]}
            
        rewrite_system_prompt = """你是美食系统中专门处理多轮对话的查询重写专家。
你的任务是：结合用户的【历史对话记录】，将用户的【最新提问】重写为一个独立、完整、且包含所有必要上下文的查询句子。

严格遵守以下规则：
1. 补全指代词：如果最新提问包含“它”、“这个”、“那个”、“另一道菜”等代词，必须根据历史对话将其替换为具体的菜名或实体。
2. 补全省略主语：如果最新提问省略了主语（例如，上文在聊“宫保鸡丁”，最新提问是“卡路里高吗？”），必须重写为“宫保鸡丁的卡路里高吗？”。
3. 保持原意：如果最新提问已经是一个完整的独立问题，且不需要任何历史背景也能看懂，请【直接原样输出】，绝不要做任何画蛇添足的修改。
4. 严禁回答问题：你只是一个翻译官！绝不允许尝试回答用户的问题！
5. 纯净输出：直接输出重写后的文本，绝不允许包含诸如“重写后：”、“查询：”或引号等任何附加格式。"""


        history_str = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history_str += f"用户说: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_str += f"美食助手说: {msg.content}\n"

        # 构造给大模型的完整 Prompt
        final_prompt = f"""
{rewrite_system_prompt}

【历史对话记录】：
{history_str}

【最新提问】：
{state['query']}"""

        llm_input = [SystemMessage(content=final_prompt)]

        response = await self.llm.ainvoke(
            input=llm_input, 
            temperature=0.1,  # 💡 重写任务需要极低的温度，保证确定性和稳定性
        )

        rewritten_query = response.content.strip()
        logger.info(f"原始查询: {state['query']} | 重写后查询: {rewritten_query}")

        return {'query': rewritten_query, 'messages': [HumanMessage(content=state['query'])]} # 不适合将重写的结果传入历史对话


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
        builder.add_node('query_rewrite_history', self.query_rewrite_history)
        builder.add_node('multi_query_composer', self.multi_query_composer)
        builder.add_node('multi_query_router', self.multi_query_router)
        builder.add_node('generate_subquery', branch_graph)
        builder.add_node('generate', self.generate_answer)

        builder.add_edge(START, 'query_rewrite_history')
        builder.add_edge('query_rewrite_history', 'multi_query_composer')
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