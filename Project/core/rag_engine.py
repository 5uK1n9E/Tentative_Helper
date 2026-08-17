"""
RAG 知识库引擎 - 基于本地文件索引的关键词检索。

不再依赖 ChromaDB 和 sentence-transformers，使用纯 Python 实现：
1. 从目录加载文档（TXT/MD/PDF/DOCX）
2. 按段落/句子分块
3. 构建倒排索引（词频统计）
4. 通过 BM25 风格的关键词匹配进行检索
"""

import os
import re
import math
import json
from typing import List, Dict, Optional
from collections import Counter

import config


# ============================================================
# 中文分词（简易版，无需外部依赖）
# ============================================================
def _tokenize(text: str) -> List[str]:
    """
    对文本进行分词。
    - 英文单词按空格/标点切分
    - 中文按字符切分（粗粒度）
    - 过滤停用词
    """
    stop_words = set("的地得的了和与或但是如果那么因为所以虽然然而可是以及"
                     "在已于被把通过对于可以及等这那其也还而")

    # 提取中文字符和英文单词
    chinese_chars = re.findall(r'[一-鿿]', text)
    english_words = re.findall(r'[a-zA-Z]+', text.lower())

    tokens = set(chinese_chars) | set(english_words)
    return sorted(tokens - stop_words)


# ============================================================
# 文档加载器
# ============================================================
def _load_text_file(file_path: str) -> List[str]:
    """加载 TXT/MD 文件，按段落返回。"""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    # 按双换行分割段落
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]  # 如果没有段落分隔，整篇作为一块
    return paragraphs


def _load_pdf_file(file_path: str) -> List[str]:
    """加载 PDF 文件（需要 pypdf）。"""
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        paragraphs = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                parts = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
                paragraphs.extend(parts if parts else [text])
        return paragraphs if paragraphs else ["[PDF 解析为空]"]
    except ImportError:
        # pypdf 未安装时，尝试用正则提取文本
        with open(file_path, "rb") as f:
            content = f.read()
        text = content.decode("utf-8", errors="ignore")
        return [text[:2000]]  # 截断


def _load_docx_file(file_path: str) -> List[str]:
    """加载 DOCX 文件（需要 python-docx）。"""
    try:
        from docx import Document
        doc = Document(file_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return paragraphs if paragraphs else ["[DOCX 解析为空]"]
    except ImportError:
        return ["[python-docx 未安装，无法解析 DOCX]"]


def load_document(file_path: str) -> List[str]:
    """
    根据文件扩展名加载文档，返回段落列表。

    Args:
        file_path: 文件完整路径

    Returns:
        段落列表
    """
    ext = os.path.splitext(file_path)[1].lower()
    loaders = {
        ".txt": _load_text_file,
        ".md": _load_text_file,
        ".pdf": _load_pdf_file,
        ".docx": _load_docx_file,
    }
    loader = loaders.get(ext)
    if not loader:
        return []
    try:
        return loader(file_path)
    except Exception as e:
        return [f"[加载失败: {str(e)}]"]


# ============================================================
# BM25 检索
# ============================================================
class BM25Index:
    """
    简易 BM25 索引和检索器。
    不依赖任何外部库，纯 Python 实现。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: 词频饱和参数
            b: 文档长度归一化参数
        """
        self.k1 = k1
        self.b = b
        self.documents: List[Dict] = []  # 每篇文档的信息
        self.doc_lengths: List[float] = []  # 文档长度（token 数）
        self.avg_doc_length: float = 0.0
        self.idf: Dict[str, float] = {}  # 逆文档频率
        self.token_counts: Dict[str, List[int]] = {}  # 每个词在各文档中的词频

    def build(self, texts: List[str], sources: List[str] = None):
        """
        构建索引。

        Args:
            texts: 文档文本列表
            sources: 文档来源标识列表（如文件名）
        """
        sources = sources or [f"doc_{i}" for i in range(len(texts))]

        # 分词
        tokenized = [_tokenize(t) for t in texts]

        # 文档长度
        self.doc_lengths = [len(toks) for toks in tokenized]
        self.avg_doc_length = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)

        # 统计词频和文档频率
        df = Counter()  # 文档频率：多少篇文档包含这个词
        for toks in tokenized:
            unique_tokens = set(toks)
            for token in unique_tokens:
                df[token] += 1

        # 计算 IDF
        N = len(texts)
        for term, doc_freq in df.items():
            # 平滑 IDF
            self.idf[term] = math.log((N - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

        # 存储每篇文档的词频
        self.token_counts = {}
        for term in df:
            freqs = []
            for toks in tokenized:
                freqs.append(toks.count(term))
            self.token_counts[term] = freqs

        # 构建文档记录
        self.documents = [
            {"text": text, "source": source, "tokens": toks}
            for text, source, toks in zip(texts, sources, tokenized)
        ]

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        检索与查询最相关的文档。

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            结果列表，每项包含 {content, source, score}
        """
        query_tokens = _tokenize(query)
        if not query_tokens or not self.documents:
            return []

        scores = []
        for i, doc in enumerate(self.documents):
            score = 0.0
            doc_len = self.doc_lengths[i]
            norm_factor = 1.0 - self.b + self.b * (doc_len / max(self.avg_doc_length, 1))

            for token in query_tokens:
                if token not in self.idf:
                    continue
                tf = self.token_counts[token][i]
                idf = self.idf[token]

                # BM25 公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * norm_factor
                score += idf * (numerator / denominator)

            scores.append({
                "content": doc["text"],
                "source": doc["source"],
                "score": score,
            })

        # 按分数降序排序
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]


# ============================================================
# RAGEngine 主类
# ============================================================
class RAGEngine:
    """RAG 知识库引擎，基于 BM25 关键词检索。"""

    LOADER_MAP = {
        ".txt": True,
        ".md": True,
        ".pdf": True,
        ".docx": True,
    }

    def __init__(self):
        """初始化引擎。"""
        self.index = BM25Index()
        self._document_registry: List[Dict] = []  # 本地记录已导入文档

    def ingest_document(self, file_path: str) -> Dict:
        """
        导入单个文档到知识库。

        Args:
            file_path: 文件路径

        Returns:
            {filename, chunks_count, success, error}
        """
        filename = os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower()

        if ext not in self.LOADER_MAP:
            return {
                "filename": filename,
                "chunks_count": 0,
                "success": False,
                "error": f"不支持的格式: {ext}",
            }

        try:
            paragraphs = load_document(file_path)
            if not paragraphs or all(not p.strip() for p in paragraphs):
                return {
                    "filename": filename,
                    "chunks_count": 0,
                    "success": False,
                    "error": "文档为空或无法解析",
                }

            # 追加到索引
            old_count = len(self.index.documents)
            self.index.build(
                [d["text"] for d in self.index.documents] + paragraphs,
                [d["source"] for d in self.index.documents] + [filename] * len(paragraphs),
            )

            # 更新注册记录
            self._document_registry.append({
                "filename": filename,
                "filepath": file_path,
                "chunks_count": len(paragraphs),
            })

            return {
                "filename": filename,
                "chunks_count": len(paragraphs),
                "success": True,
                "error": None,
            }

        except Exception as e:
            return {
                "filename": filename,
                "chunks_count": 0,
                "success": False,
                "error": str(e),
            }

    def ingest_documents(self, directory: Optional[str] = None) -> List[Dict]:
        """批量导入目录下的所有文档。"""
        directory = directory or config.KNOWLEDGE_BASE_DIR
        results = []
        if not os.path.isdir(directory):
            return results

        for filename in sorted(os.listdir(directory)):
            file_path = os.path.join(directory, filename)
            if not os.path.isfile(file_path):
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext not in self.LOADER_MAP:
                continue
            results.append(self.ingest_document(file_path))
        return results

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        检索知识库，返回最相关的文档片段。

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            [{content, source, distance}]
        """
        top_k = top_k or config.TOP_K_RETRIEVE
        if not query.strip():
            return []

        results = self.index.search(query, top_k=top_k)

        # 转换为统一格式
        items = []
        for r in results:
            items.append({
                "content": r["content"],
                "source": r["source"],
                "distance": 1.0 / (1.0 + r["score"]),  # 分数转距离
            })
        return items

    def list_documents(self) -> List[Dict]:
        """列出已导入的文档。"""
        # 按文件名合并统计
        doc_map = {}
        for doc in self._document_registry:
            fname = doc["filename"]
            if fname in doc_map:
                doc_map[fname]["chunks_count"] += doc["chunks_count"]
            else:
                doc_map[fname] = dict(doc)
        return list(doc_map.values())

    def clear(self):
        """清空知识库。"""
        self.index = BM25Index()
        self._document_registry = []

    def get_stats(self) -> Dict:
        """获取统计信息。"""
        return {
            "total_items": len(self.index.documents),
            "document_count": len(self._document_registry),
        }
