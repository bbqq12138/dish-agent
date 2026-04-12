import asyncio

from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import tool_node
from langgraph.types import Send
from typing import Annotated, Literal
from langchain_core.messages import SystemMessage
import operator
from branchGraph import BranchState
from branchGraph import generate_branch_graph

from pydantic import BaseModel, Field
import json

class Agent:
    def __init__(self, llm):
        self.app = None
        self.llm = llm


    class AgentState(MessagesState):
        query: str
        branch_queries: list[str]
        branch_categories: list[Literal["list", "detail", "general"]]
        branch_results: Annotated[list[str], operator.add]
        result: str

    
    class MultiQueryComposer(BaseModel):
        """用于将用户的查询分解为一个或多个子查询，方便后续检索"""
        analyze: str = Field(description="你对于用户查询的拆分思考过程")
        queries: list[str] = Field(description="每个元素分别是分解后的子查询")


    async def multi_query_composer(self, state: AgentState) -> AgentState:
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
 - 用户的提问中可能包含多个问题，你需要对用户的问题进行分析并在必要的时候将其拆分成多个子问题
 - 你需要将拆分出来的子问题填入JSON对象中的 'queries' 字段，该字段类型为字符串列表

举例：
 - 用户提问：我想吃点清淡的菜，有什么推荐吗？还有宫保鸡丁怎么做？
   你需要拆分成两个子问题：[推荐清淡菜, 宫保鸡丁的做法]
 - 用户提问：宫保鸡丁和红烧茄子哪个好吃？
   你需要拆分成两个子问题：[宫保鸡丁怎么样, 红烧茄子怎么样]"""

        messages = [
            {'role': 'system', 'content': multi_query_composer_prompt}, 
            {'role': 'user', 'content': state['query']}
        ]

        response = await self.llm.ainvoke(
            input = messages, 
            temperature=0.3,
            response_format = self.MultiQueryComposer,  # 底层chat.completion.parse会自动解析
        )

        try:
            branch_queries = json.loads(response.content).get('queries', [])
        except Exception as e:
            print(response.content)
            print("多查询分解出错了")
            raise Exception(e)

        branch_queries = branch_queries[:5]     # 最多拆分成5个子查询，避免过多分支

        return {'branch_queries': branch_queries}


    def parallel_retrieval_router(self, state: AgentState):
        sends = []
        for q, category in zip(state['branch_queries'], state['branch_categories']):
            sends.append(Send('generate_subquery', BranchState(subquery=q, query_category=category)))
        return sends


    async def multi_query_router(self, state: AgentState) -> AgentState:
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
            print(class_res)
            print("分类出现问题")
            raise Exception(e)

        print(state['branch_queries'])
        print(branch_categories)

        if len(branch_categories) != len(state['branch_queries']):
            raise Exception("分类结果数量与子查询数量不一致")

        return {'branch_categories': branch_categories}


    async def generate_answer(self, state: AgentState):
        """
        生成最终回答
        """
        pass



    def create_main_graph(self):
        branch_graph = generate_branch_graph()  # 生成编译好的子图节点

        builder = StateGraph(self.AgentState)
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

        self.app = builder.compile()