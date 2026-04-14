from langgraph.graph import MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import tool_node
from typing import Literal

# !子图最后一个节点必须 return {'branch_results': [str]}

class SubGraph:
    def __init__(self, llm, llm_other=None):
        self.llm = llm  # 并发的子图节点异步共享同一llm，但是可能同一API厂商会有并发限制
        self.llm_other = llm_other  # 备用API厂商

    class BranchState(MessagesState):
        subquery: str
        query_category: Literal["list", "detail", "general"]
        branch_results: list[str]  # 子图没有这个字段，则会自动过滤掉这个字段，所以不会传入主图。因此在子图最后一节点中附加这个字段并返回


    def compile_subgraph(self) -> CompiledStateGraph:
        """供主图调用生成子图节点"""
        branch_builder = StateGraph(self.BranchState)
        branch_builder.add_node('generate_subquery', self.test_branch_node)
        branch_builder.set_entry_point('generate_subquery')
        branch_builder.set_finish_point('generate_subquery')

        return branch_builder.compile()

    async def test_branch_node(self, state: BranchState):
        """测试子图节点的调用"""
        result = f"子查询: {state['subquery']}, 查询类型: {state['query_category']}"
        print(result)
        return {'branch_results': [result]}  # 注意这里必须返回这个字段，主图才会接收到这个字段并进行合并。子图要是没有这个字段，主图就会自动过滤掉这个字段，不会传入主图
