"""
端到端测试：LLM 问答 + 作业批改 + 学习报告
需要 Ollama 运行且 qwen2.5:7b 已拉取
"""

import sys
import os
import io
import json
import time

# 修复 Windows GBK 编码下 Unicode 输出问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from ollama import Client as OllamaClient
from core.rag_engine import RAGEngine
from core.grader import Grader
from core.report_generator import ReportGenerator
from core.chat_manager import ChatManager
from db.database import Database


ollama = OllamaClient(host=config.OLLAMA_BASE_URL)
MODEL = config.MODEL_NAME

# ─── 初始化 ────────────────────────────────────────────────
print("=" * 60)
print("AI 智能助教助手 — 端到端测试")
print(f"模型：{MODEL}  |  Ollama：{config.OLLAMA_BASE_URL}")
print("=" * 60)

# 验证模型可用
try:
    resp = ollama.chat(model=MODEL, messages=[{"role": "user", "content": "你好，请用一句话回复。"}],
                        options={"num_predict": 30})
    print(f"\n[OK] 模型 {MODEL} 可用，示例回复: {resp['message']['content'][:50]}...\n")
except Exception as e:
    print(f"\n[FAIL] 模型 {MODEL} 调用失败: {e}")
    print("请确认 Ollama 已运行且模型已拉取")
    sys.exit(1)

# ─── 初始化组件 ────────────────────────────────────────────
rag = RAGEngine()
sample_file = os.path.join(os.path.dirname(__file__), "..", "knowledge_base", "documents", "sample_syllabus.txt")
sample_file = os.path.abspath(sample_file)

if os.path.exists(sample_file):
    result = rag.ingest_document(sample_file)
    print(f"知识库加载: {sample_file} → {result['chunks_count']} chunks\n")
else:
    print(f"警告: 知识库文件不存在 {sample_file}，问答测试将使用纯对话模式\n")

grader = Grader()

db = Database(db_path=":memory:")
chat = ChatManager(db=db)
rp = ReportGenerator(db=db)


# ═══════════════════════════════════════════════════════════
# 测试 1：智能问答（3 个用例）
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  测试 1：智能问答")
print("=" * 60)

qa_cases = [
    {
        "id": "QA-1",
        "question": "什么是监督学习？和无监督学习有什么区别？",
        "expected_keywords": ["监督学习", "标注", "标签"],
    },
    {
        "id": "QA-2",
        "question": "CNN的全称是什么？它主要用于哪些任务？",
        "expected_keywords": ["卷积", "CNN", "图像", "视觉"],
    },
    {
        "id": "QA-3",
        "question": "这门人工智能导论课的考核方式是什么？",
        "expected_keywords": ["考核", "期末", "平时", "项目"],
    },
]
qa_results = []

for case in qa_cases:
    print(f"\n[{case['id']}] 问题: {case['question']}")
    t0 = time.time()

    # RAG 检索
    docs = rag.retrieve(case["question"], top_k=3) if rag.get_stats()["total_items"] > 0 else []
    context = "\n".join([d["content"][:300] for d in docs]) if docs else ""

    # 构建 Prompt
    if context:
        system_msg = f"你是AI课程助教。请仅根据以下课程参考资料回答问题。\n参考资料：\n{context}"
    else:
        system_msg = "你是AI课程助教。请根据你的知识回答问题。"

    try:
        resp = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": case["question"]},
            ],
            options={"num_predict": 300, "temperature": 0.3},
        )
        answer = resp["message"]["content"]
        elapsed = time.time() - t0

        # 人工评估
        keyword_hits = [kw for kw in case["expected_keywords"] if kw.lower() in answer.lower()]
        relevance = "高" if len(keyword_hits) >= 2 else ("中" if len(keyword_hits) >= 1 else "低")

        print(f"    耗时: {elapsed:.1f}s")
        print(f"    关键词命中: {keyword_hits} ({relevance})")
        print(f"    回答: {answer[:200]}...")
        print(f"    评价: 相关性={relevance}")

        qa_results.append({
            "id": case["id"],
            "question": case["question"],
            "answer_snippet": answer[:300],
            "keywords_hit": keyword_hits,
            "relevance": relevance,
            "elapsed_s": round(elapsed, 1),
            "had_context": len(docs) > 0,
            "success": True,
        })

        # 保存到会话
        chat.save_conversation("test_session", case["question"], answer, source="qa")

    except Exception as e:
        print(f"    [FAIL] {e}")
        qa_results.append({**case, "success": False, "error": str(e)})


# ═══════════════════════════════════════════════════════════
# 测试 2：作业批改（2 个用例）
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  测试 2：作业批改")
print("=" * 60)

grade_cases = [
    {
        "id": "GR-1",
        "type": "essay",
        "rubric": "essay",
        "requirement": "请解释人工智能的发展历程，包括符号主义、连接主义和行为主义三个流派的主要观点。",
        "submission": "人工智能的发展历程主要分为三个流派。符号主义认为智能可以通过符号处理和逻辑推理实现，代表成果是专家系统。连接主义受到人脑神经元结构启发，通过人工神经网络模拟智能，深度学习是其代表。行为主义强调智能体与环境的交互，通过感知和行动来学习，强化学习是其典型方法。",
    },
    {
        "id": "GR-2",
        "type": "code",
        "rubric": "code",
        "requirement": "编写一个函数，计算列表中所有元素的平均值，处理空列表的情况。",
        "submission": "def average(nums):\n    if not nums:\n        return 0\n    total = sum(nums)\n    count = len(nums)\n    return total / count",
    },
]
grade_results = []

for case in grade_cases:
    print(f"\n[{case['id']}] 类型: {case['type']} | 评分标准: {case['rubric']}")
    print(f"    作业内容: {case['submission'][:100]}...")
    t0 = time.time()

    try:
        result = grader.grade(
            submission=case["submission"],
            rubric_key=case["rubric"],
            assignment_type=case["type"],
            assignment_requirement=case["requirement"],
        )
        elapsed = time.time() - t0
        md = result.to_markdown()
        print(f"    耗时: {elapsed:.1f}s")
        print(f"    评分: {result.score}/{result.max_score}")
        print(f"    优点: {result.strengths}")
        print(f"    改进: {result.improvements}")
        if result.bugs:
            print(f"    Bug: {result.bugs}")

        grade_results.append({
            "id": case["id"],
            "type": case["type"],
            "rubric": case["rubric"],
            "score": result.score,
            "max_score": result.max_score,
            "strengths": result.strengths[:3],
            "improvements": result.improvements[:3],
            "bugs": result.bugs[:3],
            "elapsed_s": round(elapsed, 1),
            "success": True,
        })

        chat.save_conversation("test_grade", case["submission"], md, source="grade")

    except Exception as e:
        print(f"    [FAIL] {e}")
        grade_results.append({**case, "success": False, "error": str(e)})


# ═══════════════════════════════════════════════════════════
# 测试 3：学习报告
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  测试 3：学习报告生成")
print("=" * 60)

report_result = None
try:
    msgs = db.get_all_messages("test_session")
    if msgs and len(msgs) >= 2:
        t0 = time.time()
        report = rp.generate("test_session")
        elapsed = time.time() - t0
        print(f"    耗时: {elapsed:.1f}s")
        print(f"    参与度评分: {report.engagement_score}/100")
        print(f"    话题: {report.topics[:5]}")
        print(f"    优势: {report.strengths[:5]}")
        print(f"    薄弱点: {report.weak_areas[:5]}")
        report_result = {
            "engagement_score": report.engagement_score,
            "topics": report.topics[:5],
            "strengths": report.strengths[:5],
            "weak_areas": report.weak_areas[:5],
            "elapsed_s": round(elapsed, 1),
            "success": True,
        }
    else:
        print(f"    跳过: 对话消息不足 (需要至少 2 条，实际 {len(msgs)})")
        report_result = {"success": False, "error": "消息不足"}
except Exception as e:
    print(f"    [FAIL] {e}")
    report_result = {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════
# 汇总
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  测试汇总")
print("=" * 60)

# 输出 JSON 结果文件
summary = {
    "model": MODEL,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "qa_tests": qa_results,
    "grade_tests": grade_results,
    "report_test": report_result,
}

out_path = os.path.join(os.path.dirname(__file__), "..", "test_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

qa_pass = sum(1 for r in qa_results if r.get("success"))
qa_total = len(qa_results)
gr_pass = sum(1 for r in grade_results if r.get("success"))
gr_total = len(grade_results)

print(f"问答测试: {qa_pass}/{qa_total} 通过")
print(f"批改测试: {gr_pass}/{gr_total} 通过")
print(f"报告测试: {'通过' if report_result.get('success') else '失败'}")
print(f"\n结果已保存到: {out_path}")
db.close()
