"""
RAG 引擎单元测试（BM25 版本）
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag_engine import RAGEngine


def test_rag_engine_initialization():
    """测试 RAG 引擎初始化。"""
    engine = RAGEngine()
    assert engine is not None
    assert engine.index is not None
    stats = engine.get_stats()
    assert stats["total_items"] == 0
    assert stats["document_count"] == 0
    print("PASS  RAG 引擎初始化测试通过")


def test_ingest_txt_document():
    """测试导入 TXT 文档。"""
    engine = RAGEngine()
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("人工智能是计算机科学的一个分支，它致力于理解智能的本质。")

        result = engine.ingest_document(test_file)
        assert result["success"] is True
        assert result["chunks_count"] > 0
        print(f"PASS  TXT 导入测试通过，生成 {result['chunks_count']} 个片段")


def test_retrieve():
    """测试检索功能。"""
    engine = RAGEngine()
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("机器学习是人工智能的核心技术之一。深度学习是机器学习的一个重要分支。")

        engine.ingest_document(test_file)

        results = engine.retrieve("什么是深度学习？", top_k=1)
        assert len(results) > 0
        assert "深度学习" in results[0]["content"]
        print(f"PASS  检索测试通过，返回 {len(results)} 条结果")


def test_unsupported_format():
    """测试不支持的文件格式。"""
    engine = RAGEngine()
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.exe")
        with open(test_file, "w") as f:
            f.write("fake")

        result = engine.ingest_document(test_file)
        assert result["success"] is False
        assert "不支持" in result["error"]
        print("PASS  不支持格式测试通过")


def test_retrieve_empty():
    """测试空知识库下的检索。"""
    engine = RAGEngine()
    results = engine.retrieve("随便问什么")
    assert results == []
    print("PASS  空库检索测试通过")


def test_multiple_ingest():
    """测试多次导入文档。"""
    engine = RAGEngine()
    with tempfile.TemporaryDirectory() as tmpdir:
        f1 = os.path.join(tmpdir, "doc1.txt")
        with open(f1, "w", encoding="utf-8") as f:
            f.write("监督学习需要标注数据。")

        f2 = os.path.join(tmpdir, "doc2.txt")
        with open(f2, "w", encoding="utf-8") as f:
            f.write("无监督学习不需要标注数据。")

        r1 = engine.ingest_document(f1)
        assert r1["success"]
        r2 = engine.ingest_document(f2)
        assert r2["success"]

        stats = engine.get_stats()
        assert stats["document_count"] == 2

        docs = engine.list_documents()
        assert len(docs) == 2
    print(f"PASS  多次导入测试通过，文档数：{len(docs)}")


if __name__ == "__main__":
    test_rag_engine_initialization()
    test_ingest_txt_document()
    test_retrieve()
    test_unsupported_format()
    test_retrieve_empty()
    test_multiple_ingest()
    print("\n所有 RAG 测试通过！")
