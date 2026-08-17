"""
LangChain Chain 工厂 - 将 Prompt 模板、LLM 和工具链组合成可执行的 Chain。
"""

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import config
from llm.model_client import get_llm, get_parser
from llm import prompt_templates


def build_qa_chain(retriever=None):
    """
    构建智能问答 Chain。

    流程: 用户问题 -> 检索知识库 -> 组装 Prompt -> LLM 生成答案

    Args:
        retriever: ChromaDB retriever 实例，如果不提供则使用默认检索

    Returns:
        LangChain Runnable Chain
    """
    llm = get_llm(streaming=True)

    if retriever:
        # RAG 模式：先检索再问答
        def format_docs(docs):
            return "\n\n".join([f"[文档{i+1}] {doc.page_content}" for i, doc in enumerate(docs)])

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_templates.qa_prompt
            | llm
            | get_parser()
        )
        return rag_chain
    else:
        # 普通对话模式
        chain = (
            prompt_templates.qa_prompt
            | llm
            | get_parser()
        )
        return chain


def build_grading_chain():
    """
    构建作业批改 Chain。

    Returns:
        LangChain Runnable Chain
    """
    llm = get_llm(temperature=config.MODEL_TEMPERATURE)

    chain = (
        prompt_templates.grading_prompt
        | llm
        | get_parser()
    )
    return chain


def build_code_grading_chain():
    """
    构建代码批改 Chain。

    Returns:
        LangChain Runnable Chain
    """
    llm = get_llm(temperature=config.MODEL_TEMPERATURE)

    chain = (
        prompt_templates.code_grading_prompt
        | llm
        | get_parser()
    )
    return chain


def build_report_chain():
    """
    构建学习报告生成 Chain。

    Returns:
        LangChain Runnable Chain
    """
    llm = get_llm(temperature=config.MODEL_TEMPERATURE)

    chain = (
        prompt_templates.report_prompt
        | llm
        | get_parser()
    )
    return chain


def build_simple_chat_chain():
    """
    构建简单对话 Chain（不依赖检索或批改）。

    Returns:
        LangChain Runnable Chain
    """
    llm = get_llm(streaming=True)

    chain = (
        prompt_templates.qa_prompt
        | llm
        | get_parser()
    )
    return chain
