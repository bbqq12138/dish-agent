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

def test_branch_node(state: BranchState):
    """测试子图节点的调用"""
    result = f"子查询: {state['subquery']}, 查询类型: {state['query_category']}"
    print(result)
