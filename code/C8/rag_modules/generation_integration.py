"""
生成集成模块
"""

import json
import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

class GenerationIntegrationModule:
    """生成集成模块 - 负责LLM集成和回答生成"""
    
    def __init__(self, llm):
        """
        初始化生成集成模块
        
        Args:
            llm: 大语言模型实例
        """
        self.llm = llm
    
    async def generate_basic_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成基础回答

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            生成的回答
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪助手。请根据以下食谱信息回答用户的问题。

用户问题: {question}

相关食谱信息:
{context}

请提供详细、实用的回答。如果信息不足，请诚实说明。

回答:""")

        # 使用LCEL构建链
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response
    
    async def generate_step_by_step_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成分步骤回答

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            分步骤的详细回答
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪导师。请根据食谱信息和用户请求，为用户提供详细的分步骤指导。

用户问题: {question}

相关食谱信息:
{context}

请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

## 🥘 菜品介绍
[简要介绍菜品特点和难度]

## 🛒 所需食材
[列出主要食材和用量]

## 👨‍🍳 制作步骤
[详细的分步骤说明，每步包含具体操作和大概所需时间]

## 💡 制作技巧
[仅在有实用技巧时包含。优先使用原文中的实用技巧，如果原文的"附加内容"与烹饪无关或为空，可以基于制作步骤总结关键要点，或者完全省略此部分]

注意：
- 如果没有相关食谱信息，请直接回答 "抱歉，没有找到相关食谱信息，无法提供详细指导。请更改描述或尝试其他查询。"
- 根据实际内容和用户请求灵活调整结构，回答要满足用户需求，但不需要为了满足结构而填充无关内容
- 如：用户只需要食材那就只输出食材部分，用户需要制作步骤那就全部输出

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = await chain.ainvoke(query)
        return response
    
    def query_rewrite(self, query: str) -> str:
        """
        智能查询重写 - 让大模型判断是否需要重写查询

        Args:
            query: 原始查询

        Returns:
            重写后的查询或原查询
        """
        prompt = PromptTemplate(
            template="""
你是一个智能查询分析助手。请分析用户的查询，判断是否需要重写以提高食谱搜索效果。

原始查询: {query}

分析规则：
1. **具体明确的查询**（直接返回原查询）：
   - 包含具体菜品名称：如"宫保鸡丁怎么做"、"红烧肉的制作方法"
   - 明确的制作询问：如"蛋炒饭需要什么食材"、"糖醋排骨的步骤"
   - 具体的烹饪技巧：如"如何炒菜不粘锅"、"怎样调制糖醋汁"

2. **模糊不清的查询**（需要重写）：
   - 过于宽泛：如"做菜"、"有什么好吃的"、"推荐个菜"
   - 缺乏具体信息：如"川菜"、"素菜"、"简单的"
   - 口语化表达：如"想吃点什么"、"有饮品推荐吗"

重写原则：
- 保持原意不变
- 增加相关烹饪术语
- 优先推荐简单易做的
- 保持简洁性

示例：
- "做菜" → "简单易做的家常菜谱"
- "有饮品推荐吗" → "简单饮品制作方法"
- "推荐个菜" → "简单家常菜推荐"
- "川菜" → "经典川菜菜谱"
- "宫保鸡丁怎么做" → "宫保鸡丁怎么做"（保持原查询）
- "红烧肉需要什么食材" → "红烧肉需要什么食材"（保持原查询）

请输出最终查询（如果不需要重写就返回原查询）:""",
            input_variables=["query"]
        )

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query).strip()

        # 记录重写结果
        if response != query:
            logger.info(f"查询已重写: '{query}' → '{response}'")
        else:
            logger.info(f"查询无需重写: '{query}'")

        return response



    def query_router(self, query: str) -> str:
        """
        查询路由 - 根据查询类型选择不同的处理方式

        Args:
            query: 用户查询

        Returns:
            路由类型 ('list', 'detail', 'general')
        """
        
        prompt = ChatPromptTemplate.from_template("""
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

    async def generate_list_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成列表式回答 - 适用于推荐类查询

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            列表式回答
        """

        # dish_context = "\n\n".join(list(doc.page_content for doc in context_docs))
        context = self._build_context(context_docs)

        list_answer_prompt = f"""你是一个菜品推荐助手。根据用户的提问和相关的菜品资料，生成一个简洁的推荐列表回答。

格式：
1. 菜品名称，理由
2. 菜品名称，理由
...


要求：
 - 请严格按照上述格式生成回答
 - 根据用户的要求和资料选择其中最相关的菜品，推荐理由可以简短说明，当不需要理由时可以省略
 - 当没有与用户要求相关的菜品信息时，请直接返回 "抱歉，没有找到相关的菜品信息，无法提供推荐。请更改描述或尝试其他查询。"

相关菜品信息:
{context}

用户问题: {query}"""
        
        result = await self.llm.ainvoke(
            input=[SystemMessage(content=list_answer_prompt)], 
            temperature=0.6, 
        )

        return result.content.strip()


    async def generate_list_answer_stream(self, query: str, context_docs: List[Document]):
        """
        生成列表式回答 - 适用于推荐类查询

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            列表式回答
        """

        context = self._build_context(context_docs)
        list_answer_prompt = f"""你是一个菜品推荐助手。根据用户的提问和相关的菜品资料，生成一个简洁的推荐列表回答。

格式：
1. 菜品名称，理由
2. 菜品名称，理由
...


要求：
 - 请严格按照上述格式生成回答
 - 根据用户的要求和资料选择其中最相关的菜品，推荐理由可以简短说明，当不需要理由时可以省略
 - 当没有与用户要求相关的菜品信息时，请直接返回 "抱歉，没有找到相关的菜品信息，无法提供推荐。请更改描述或尝试其他查询。"

相关菜品信息:
{context}

用户问题: {query}"""
        

        stream_iter = self.llm.astream(
            input=[SystemMessage(content=list_answer_prompt)], 
            temperature=0.6,
        )

        async for output_chunk in stream_iter:
            yield output_chunk.content      # yield 返回生成器而不是直接返回值或返回协程，因此 async 函数使用yield，返回的是异步生成器


    async def generate_basic_answer_stream(self, query: str, context_docs: List[Document]):
        """
        生成基础回答 - 流式输出

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Yields:
            生成的回答片段
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪助手。请根据以下食谱信息回答用户的问题。

用户问题: {question}

相关食谱信息:
{context}

请提供详细、实用的回答。如果信息不足，请诚实说明。

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        async for chunk in chain.astream(query):
            yield chunk

    async def generate_step_by_step_answer_stream(self, query: str, context_docs: List[Document]):
        """
        生成详细步骤回答 - 流式输出

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Yields:
            详细步骤回答片段
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪导师。请根据食谱信息和用户请求，为用户提供详细的分步骤指导。

用户问题: {question}

相关食谱信息:
{context}

请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

## 🥘 菜品介绍
[简要介绍菜品特点和难度]

## 🛒 所需食材
[列出主要食材和用量]

## 👨‍🍳 制作步骤
[详细的分步骤说明，每步包含具体操作和大概所需时间]

## 💡 制作技巧
[仅在有实用技巧时包含。如果原文的"附加内容"与烹饪无关或为空，可以基于制作步骤总结关键要点，或者完全省略此部分]

注意：
- 如果没有相关食谱信息，请直接回答 "抱歉，没有找到相关食谱信息，无法提供详细指导。请更改描述或尝试其他查询。"
- 根据实际内容和用户请求灵活调整结构，回答要满足用户需求，但不需要为了满足结构而填充无关内容
- 如：用户只需要食材那就只输出食材部分，用户需要制作步骤那就全部输出

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        async for chunk in chain.astream(query):
            yield chunk

    def _build_context(self, docs: List[Document], max_length: int = 5000) -> str:
        """
        构建上下文字符串
        
        Args:
            docs: 文档列表
            max_length: 最大长度，超过后会截断，2000大概两个食谱，设置5000可以保证大概4-5个食谱的完整信息
            
        Returns:
            格式化的上下文字符串
        """
        if not docs:
            return "暂无相关食谱信息。"
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(docs, 1):
            # 添加元数据信息
            metadata_info = f"【食谱 {i}】"
            if 'dish_name' in doc.metadata:
                metadata_info += f" {doc.metadata['dish_name']}"
            if 'category' in doc.metadata:
                metadata_info += f" | 分类: {doc.metadata['category']}"
            if 'difficulty' in doc.metadata:
                metadata_info += f" | 难度: {doc.metadata['difficulty']}"
            
            # 构建文档文本
            doc_text = f"{metadata_info}\n{doc.page_content}\n"
            
            # 检查长度限制
            if current_length + len(doc_text) > max_length:
                logger.info(f"达到上下文长度限制，已添加 {i-1} 条食谱信息")
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n" + "="*50 + "\n".join(context_parts)
