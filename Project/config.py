"""
全局配置文件 - AI 智能助教助手
所有路径、模型名称、超参数集中管理。
"""

import os

# ==================== 基础路径 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================== Ollama 配置 ====================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:7b")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
MODEL_TEMPERATURE = 0.3
MAX_TOKENS = 2048

# ==================== 路径配置 ====================
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
DB_PATH = os.path.join(BASE_DIR, "teaching_assistant.db")
SCHEMA_SQL = os.path.join(BASE_DIR, "db", "schema.sql")
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "knowledge_base", "documents")
RUBRICS_FILE = os.path.join(BASE_DIR, "rubrics.json")

# ==================== RAG 参数 ====================
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RETRIEVE = 5

# ==================== 支持的文档格式 ====================
SUPPORTED_EXTENSIONS = [".txt", ".md", ".pdf", ".docx"]

# ==================== 会话配置 ====================
SESSION_HISTORY_LIMIT = 50  # 每个会话最多保留的消息数


def ensure_dirs():
    """确保所有需要的目录存在。"""
    os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)


ensure_dirs()
