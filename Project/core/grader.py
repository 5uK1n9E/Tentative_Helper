"""
作业批改引擎 - 支持文本作业和代码作业的自动批改。

功能：
1. 文本作业批改（作文、简答、论述等）
2. 代码作业批改（正确性、风格、效率）
3. 多维度评分（基于评分标准）
4. 详细反馈生成（优点 + 改进建议）
"""

import json
import os
from typing import Optional, Dict, List
from dataclasses import dataclass, field, asdict

import config
from ollama import Client as OllamaClient

_ollama_client = OllamaClient(host=config.OLLAMA_BASE_URL)


@dataclass
class GradeResult:
    """批改结果数据结构。"""
    score: int = 0
    max_score: int = 10
    feedback: str = ""
    strengths: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)
    bugs: List[str] = field(default_factory=list)  # 仅代码批改

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        """格式化为 Markdown 字符串。"""
        lines = [
            f"## 评分结果",
            "",
            f"**得分：{self.score} / {self.max_score}**",
            "",
        ]
        if self.strengths:
            lines.append("### 优点")
            for s in self.strengths:
                lines.append(f"- {s}")
            lines.append("")
        if self.improvements:
            lines.append("### 改进建议")
            for imp in self.improvements:
                lines.append(f"- {imp}")
            lines.append("")
        if self.bugs:
            lines.append("### 发现的 Bug")
            for bug in self.bugs:
                lines.append(f"- {bug}")
            lines.append("")
        if self.feedback:
            lines.append("### 详细反馈")
            lines.append(self.feedback)
            lines.append("")
        return "\n".join(lines)


class Grader:
    """作业批改引擎。"""

    def __init__(self, rubrics_file: Optional[str] = None):
        """
        Args:
            rubrics_file: 评分标准 JSON 文件路径
        """
        self.rubrics_file = rubrics_file or config.RUBRICS_FILE
        self.rubrics = self._load_rubrics()

    def _load_rubrics(self) -> dict:
        """加载评分标准。"""
        if os.path.exists(self.rubrics_file):
            with open(self.rubrics_file, "r", encoding="utf-8") as f:
                return json.load(f)
        # 如果文件不存在，创建默认评分标准
        default_rubrics = self._default_rubrics()
        self._save_rubrics(default_rubrics)
        return default_rubrics

    def _default_rubrics(self) -> dict:
        """生成默认评分标准。"""
        return {
            "essay": {
                "name": "作文/简答题",
                "max_score": 10,
                "criteria": [
                    {"name": "内容准确性", "weight": 5, "description": "答案是否正确、事实是否准确"},
                    {"name": "逻辑结构", "weight": 3, "description": "回答是否有条理、逻辑是否清晰"},
                    {"name": "语言表达", "weight": 2, "description": "语言是否通顺、表达是否准确"},
                ],
            },
            "code": {
                "name": "编程作业",
                "max_score": 10,
                "criteria": [
                    {"name": "功能正确性", "weight": 5, "description": "代码是否能正确运行并达到预期效果"},
                    {"name": "代码规范", "weight": 3, "description": "代码风格是否符合规范、变量命名是否清晰"},
                    {"name": "算法效率", "weight": 2, "description": "算法的时间复杂度和空间复杂度是否合理"},
                ],
            },
            "project": {
                "name": "项目作业",
                "max_score": 10,
                "criteria": [
                    {"name": "功能完整性", "weight": 4, "description": "是否实现了所有要求的功能"},
                    {"name": "技术深度", "weight": 3, "description": "使用的技术是否体现了课程所学知识"},
                    {"name": "创新性与文档", "weight": 3, "description": "是否有创新点、文档是否完善"},
                ],
            },
        }

    def _save_rubrics(self, rubrics: dict):
        """保存评分标准到文件。"""
        with open(self.rubrics_file, "w", encoding="utf-8") as f:
            json.dump(rubrics, f, ensure_ascii=False, indent=2)

    def grade(self, submission: str, rubric_key: str = "essay",
              assignment_type: str = "text",
              assignment_requirement: str = "") -> GradeResult:
        """
        批改一份作业。

        Args:
            submission: 学生提交的作业内容
            rubric_key: 评分标准键名（essay / code / project）
            assignment_type: 作业类型（text / code）
            assignment_requirement: 题目要求（代码批改时必填）

        Returns:
            GradeResult 对象
        """
        if rubric_key not in self.rubrics:
            raise ValueError(f"未知的评分标准: {rubric_key}，可用的有: {list(self.rubrics.keys())}")

        rubric = self.rubrics[rubric_key]
        max_score = rubric["max_score"]

        # 构建评分标准描述
        criteria_desc = "\n".join(
            f"- {c['name']}（{c['weight']}分）：{c['description']}"
            for c in rubric["criteria"]
        )

        # 根据类型选择 Prompt
        if assignment_type == "code":
            prompt = self._build_code_grade_prompt(
                submission, criteria_desc, max_score, assignment_requirement
            )
        else:
            prompt = self._build_text_grade_prompt(
                submission, criteria_desc, max_score, rubric["name"]
            )

        # 调用 Ollama 进行批改
        try:
            response = _ollama_client.chat(
                model=config.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": 1024, "temperature": 0.3},
            )
            raw_text = response["message"]["content"]
        except Exception as e:
            return GradeResult(
                score=0,
                max_score=max_score,
                feedback=f"批改失败：{str(e)}",
                strengths=[],
                improvements=["请检查 Ollama 服务是否正常运行"],
            )

        # 解析 JSON 结果
        return self._parse_grade_response(raw_text, max_score, assignment_type == "code")

    def _build_text_grade_prompt(self, submission: str, criteria_desc: str,
                                  max_score: int, rubric_name: str) -> str:
        """构建文本作业批改 Prompt。"""
        return (
            f"你是一位严格的课程助教，请按照以下评分标准批改这份作业。\n\n"
            f"【评分标准：{rubric_name}】\n{criteria_desc}\n"
            f"满分：{max_score} 分\n\n"
            f"【学生提交内容】\n{submission}\n\n"
            f"请严格按以下 JSON 格式返回评分结果（不要添加任何其他文字）：\n"
            f"{{\"score\": 分数(整数), \"feedback\": \"详细反馈\", "
            f"\"strengths\": [\"优点1\", \"优点2\"], \"improvements\": [\"改进建议1\", \"改进建议2\"]}}"
        )

    def _build_code_grade_prompt(self, submission: str, criteria_desc: str,
                                  max_score: int, requirement: str) -> str:
        """构建代码作业批改 Prompt。"""
        req_text = f"\n【题目要求】\n{requirement}" if requirement else ""
        return (
            f"你是一位编程课程助教，请按照以下评分标准批改这份代码作业。\n\n"
            f"【评分标准】\n{criteria_desc}\n"
            f"满分：{max_score} 分\n"
            f"{req_text}\n\n"
            f"【学生代码】\n```python\n{submission}\n```\n\n"
            f"请严格按以下 JSON 格式返回评分结果（不要添加任何其他文字）：\n"
            f"{{\"score\": 分数(整数), \"feedback\": \"详细反馈\", "
            f"\"strengths\": [\"优点1\", \"优点2\"], \"improvements\": [\"改进建议1\"], "
            f"\"bugs\": [\"bug描述1\"]}}"
        )

    def _parse_grade_response(self, raw_text: str, max_score: int,
                               is_code: bool = False) -> GradeResult:
        """解析 LLM 返回的 JSON 评分结果。"""
        # 尝试提取 JSON 代码块
        json_str = raw_text.strip()
        if "```" in json_str:
            # 提取 ```json ... ``` 或 ``` ... ``` 中的内容
            parts = json_str.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    json_str = part
                    break

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # 如果解析失败，返回默认结果
            return GradeResult(
                score=0,
                max_score=max_score,
                feedback=f"评分解析失败。LLM 返回：{raw_text[:500]}",
                strengths=[],
                improvements=["请重试或手动批改"],
            )

        strengths = data.get("strengths", [])
        improvements = data.get("improvements", [])
        bugs = data.get("bugs", []) if is_code else []

        return GradeResult(
            score=data.get("score", 0),
            max_score=max_score,
            feedback=data.get("feedback", "暂无反馈"),
            strengths=strengths if isinstance(strengths, list) else [strengths],
            improvements=improvements if isinstance(improvements, list) else [improvements],
            bugs=bugs if isinstance(bugs, list) else [],
        )

    def list_rubrics(self) -> List[Dict]:
        """列出所有可用的评分标准。"""
        results = []
        for key, rubric in self.rubrics.items():
            results.append({
                "key": key,
                "name": rubric.get("name", key),
                "max_score": rubric["max_score"],
                "criteria": rubric.get("criteria", []),
            })
        return results
