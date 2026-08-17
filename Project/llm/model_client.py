"""
Ollama 模型客户端封装 - 通过 LangChain 连接本地 Ollama 服务。
支持流式输出和非流式输出两种模式。
"""

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from typing import Optional

import config


def get_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    streaming: bool = False,
) -> ChatOllama:
    """
    获取配置的 Ollama LLM 实例。

    Args:
        model_name: 模型名称，默认使用 config.MODEL_NAME
        temperature: 采样温度，默认使用 config.MODEL_TEMPERATURE
        streaming: 是否启用流式输出

    Returns:
        配置好的 ChatOllama 实例
    """
    model = model_name or config.MODEL_NAME
    temp = temperature if temperature is not None else config.MODEL_TEMPERATURE

    llm = ChatOllama(
        base_url=config.OLLAMA_BASE_URL,
        model=model,
        temperature=temp,
        num_predict=config.MAX_TOKENS,
        stream=streaming,
    )
    return llm


def get_text_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
) -> ChatOllama:
    """
    获取文本生成专用 LLM（temperature=0 保证确定性输出，适用于批改场景）。

    Args:
        model_name: 模型名称
        temperature: 采样温度

    Returns:
        配置好的 ChatOllama 实例
    """
    model = model_name or config.MODEL_NAME
    temp = temperature if temperature is not None else 0.0

    llm = ChatOllama(
        base_url=config.OLLAMA_BASE_URL,
        model=model,
        temperature=temp,
        num_predict=config.MAX_TOKENS,
    )
    return llm


def get_parser() -> StrOutputParser:
    """返回文本输出解析器。"""
    return StrOutputParser()
