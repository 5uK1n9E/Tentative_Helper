"""
作业批改引擎单元测试
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.grader import Grader, GradeResult


def test_grader_initialization():
    """测试批改器初始化。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        rubrics_file = os.path.join(tmpdir, "rubrics.json")
        grader = Grader(rubrics_file=rubrics_file)
        assert grader is not None
        assert "essay" in grader.rubrics
        print("✅ 批改器初始化测试通过")


def test_list_rubrics():
    """测试列出评分标准。"""
    grader = Grader()
    rubrics = grader.list_rubrics()
    assert len(rubrics) > 0
    assert any(r["key"] == "essay" for r in rubrics)
    print(f"✅ 评分标准列表测试通过，共 {len(rubrics)} 个标准")


def test_parse_grade_json():
    """测试解析评分 JSON 结果。"""
    grader = Grader()

    # 模拟 LLM 返回的合法 JSON
    mock_response = json.dumps({
        "score": 8,
        "max_score": 10,
        "feedback": "回答得很好，但逻辑结构可以进一步优化。",
        "strengths": ["内容准确", "观点明确"],
        "improvements": ["加强段落衔接", "补充论据"],
    })

    result = grader._parse_grade_response(mock_response, 10, is_code=False)
    assert isinstance(result, GradeResult)
    assert result.score == 8
    assert result.max_score == 10
    assert len(result.strengths) == 2
    print("✅ JSON 解析测试通过")


def test_parse_invalid_json():
    """测试解析非法 JSON 的容错处理。"""
    grader = Grader()

    mock_response = "这不是有效的 JSON 格式！随便说点什么..."

    result = grader._parse_grade_response(mock_response, 10, is_code=False)
    assert isinstance(result, GradeResult)
    assert result.score == 0
    assert "评分解析失败" in result.feedback
    print("✅ 非法 JSON 容错测试通过")


def test_grade_result_to_markdown():
    """测试 GradeResult 的 Markdown 格式化。"""
    result = GradeResult(
        score=8,
        max_score=10,
        feedback="整体表现不错",
        strengths=["论点清晰"],
        improvements=["加强论证"],
    )

    md = result.to_markdown()
    assert "8 / 10" in md
    assert "论点清晰" in md
    assert "加强论证" in md
    print("✅ Markdown 格式化测试通过")


if __name__ == "__main__":
    test_grader_initialization()
    test_list_rubrics()
    test_parse_grade_json()
    test_parse_invalid_json()
    test_grade_result_to_markdown()
    print("\n🎉 所有批改测试通过！")
