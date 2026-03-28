from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from typing import Literal

# !子图最后一个节点必须 return {'branch_results': [str]}，返回类型AgentState

class BranchState(MessagesState):
    subquery: str
    query_category: Literal["list", "detail", "general"]
    result: str     # 最后一个子查询节点要返回 {'branch_results': [result]}直接作为主节点键branch_results的附加值


def generate_branch_graph() -> CompiledStateGraph:
    """供主图调用生成子图节点"""
    branch_builder = StateGraph(BranchState)

    return branch_builder.compile()


async def query_router(state: BranchState) -> BranchState:
    """
    查询路由 - 根据查询类型选择不同的处理方式

    Args:
        query: 用户查询

    Returns:
        路由类型 ('list', 'detail', 'general')
    """
    prompt = ChatPromptTemplate.from_template("""
根据用户的问题，将其分类为以下三种类型之一：

1. 'list' - 用户想要获取菜品列表或推荐，只需要菜名
   例如：推荐几个素菜、有什么川菜、给我3个简单的菜

2. 'detail' - 用户想要具体的制作方法或详细信息
   例如：宫保鸡丁怎么做、制作步骤、需要什么食材

3. 'general' - 其他一般性问题
   例如：什么是川菜、制作技巧、营养价值

请只返回分类结果：list、detail 或 general

用户问题: {query}

分类结果:""")

    chain = (
        {"query": RunnablePassthrough()}
        | prompt
        | self.llm
        | StrOutputParser()
    )

    result = chain.invoke(query).strip().lower()

    # 确保返回有效的路由类型
    if result in ['list', 'detail', 'general']:
        return result
    else:
        return 'general'  # 默认类型