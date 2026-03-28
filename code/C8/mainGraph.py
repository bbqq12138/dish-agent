import asyncio
import random

from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import tool_node
from langgraph.types import Send
from typing import Annotated, Literal
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
import operator
from branchGraph import BranchState
from branchGraph import generate_branch_graph


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

用户的提问中可能包含多个问题，你需要对用户的问题进行分析并在必要的时候将其拆分成多个子问题。

如：
- 用户提问：我想吃点清淡的菜，有什么推荐吗？还有宫保鸡丁怎么做？
- 你需要拆分成两个子问题：1.推荐清淡菜 2.宫保鸡丁的做法
- 用户提问：宫保鸡丁和红烧茄子哪个好吃？
- 你需要拆分成两个子问题：1.宫保鸡丁怎么样 2.红烧茄子怎么样

请直接将拆分后的多个子问题返回，以空格分隔，如：推荐清淡菜 宫保鸡丁的做法

用户问题：{state['query']}

子问题:"""

        result = await self.llm.ainvoke(([SystemMessage(content=multi_query_composer_prompt)]))
        try:
            branch_queries = list(result.content.split(' '))
        except Exception as e:
            print(result)
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
请根据用户的多个问题，将其依次分类为以下三种类型之一：

1. 'list' - 用户想要获取菜品推荐或列表
   例如：推荐几个素菜、有什么川菜、高血压适合吃什么菜、夜晚看球赛适合搭配什么菜

2. 'detail' - 用户想要与菜品制作相关信息
   例如：宫保鸡丁怎么做、制作步骤、需要什么食材

3. 'general' - 针对菜品的其他一般性问题
   例如：什么是川菜、菜品口味热量等特性

只返回分类结果：list、detail 或 general
若有多个查询，则以空格分隔，如: list detail list

用户问题: {'  '.join(state['branch_queries'])}

分类结果:"""
        class_res = await self.llm.ainvoke([SystemMessage(content=query_router_prompt)])
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
        branch_graph = generate_branch_graph()  # 生成子图节点

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