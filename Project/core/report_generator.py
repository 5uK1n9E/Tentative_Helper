"""
学习报告生成器 - 基于对话历史生成个性化学习分析报告。

功能：
1. 从对话历史中提取讨论的话题
2. 分析学生的知识优势和薄弱点
3. 生成个性化学习建议
4. 计算综合参与度评分
"""

import json
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict

import config
from db.database import Database
from ollama import Client as OllamaClient

_ollama_client = OllamaClient(host=config.OLLAMA_BASE_URL)


@dataclass
class LearningReport:
    """学习报告数据结构。"""
    session_id: str = ""
    total_interactions: int = 0
    topics: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    weak_areas: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    engagement_score: float = 0.0
    generated_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        """格式化为 Markdown 字符串。"""
        lines = [
            "# 学习分析报告",
            "",
            f"**会话 ID：** {self.session_id}",
            f"**交互次数：** {self.total_interactions} 次",
            f"**生成时间：** {self.generated_at}",
            f"**综合参与度：** {self.engagement_score:.0f} / 100",
            "",
        ]

        if self.topics:
            lines.append("## 讨论话题")
            lines.append(", ".join(f"`{t}`" for t in self.topics))
            lines.append("")

        if self.strengths:
            lines.append("## 知识优势")
            for s in self.strengths:
                lines.append(f"- {s}")
            lines.append("")

        if self.weak_areas:
            lines.append("## 待加强领域")
            for w in self.weak_areas:
                lines.append(f"- {w}")
            lines.append("")

        if self.suggestions:
            lines.append("## 学习建议")
            for s in self.suggestions:
                lines.append(f"- {s}")
            lines.append("")

        return "\n".join(lines)


class ReportGenerator:
    """学习报告生成器。"""

    def __init__(self, db: Optional[Database] = None):
        """
        Args:
            db: Database 实例，如果不提供则自动创建
        """
        self.db = db or Database()

    def generate(self, session_id: str) -> LearningReport:
        """
        为指定会话生成学习分析报告。

        Args:
            session_id: 会话 ID

        Returns:
            LearningReport 对象
        """
        # 获取全部对话历史
        messages = self.db.get_all_messages(session_id)

        if not messages:
            return LearningReport(
                session_id=session_id,
                total_interactions=0,
                engagement_score=0.0,
                generated_at=config.__dict__.get("_", ""),
            )

        # 构建历史摘要
        history_text = self._build_history_summary(messages)

        # 调用 LLM 生成报告
        prompt = self._build_report_prompt(history_text)

        try:
            response = _ollama_client.chat(
                model=config.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": 1500, "temperature": 0.3},
            )
            raw_text = response["message"]["content"]
        except Exception as e:
            return LearningReport(
                session_id=session_id,
                total_interactions=len(messages),
                strengths=["无法生成报告"],
                weak_areas=[str(e)],
                engagement_score=0.0,
            )

        # 解析结果
        report = self._parse_report_response(raw_text, session_id, messages)
        return report

    def _build_history_summary(self, messages: List[Dict]) -> str:
        """
        将对话历史压缩为文本摘要。
        每条消息取前 300 字符以避免上下文过长。
        """
        lines = []
        for msg in messages:
            role_label = {"user": "学生", "assistant": "助教", "system": "系统"}.get(
                msg.get("role", ""), "未知"
            )
            content = msg.get("content", "")[:300]
            lines.append(f"[{role_label}]: {content}")
        return "\n".join(lines)

    def _build_report_prompt(self, history_text: str) -> str:
        """构建报告生成 Prompt。"""
        return (
            f"你是一位学习分析专家。请根据以下学生的学习对话历史，"
            f"生成一份个性化的学习分析报告。\n\n"
            f"【对话历史】\n{history_text}\n\n"
            f"请严格按以下 JSON 格式返回（不要添加任何其他文字）：\n"
            f"{{"
            f"\"topics\": [\"话题1\", \"话题2\"], "
            f"\"strengths\": [\"优势1\", \"优势2\"], "
            f"\"weak_areas\": [\"薄弱点1\", \"薄弱点2\"], "
            f"\"suggestions\": [\"建议1\", \"建议2\"], "
            f"\"engagement_score\": 参与度评分(0-100的整数)"
            f"}}"
        )

    def _parse_report_response(self, raw_text: str, session_id: str,
                                messages: List[Dict]) -> LearningReport:
        """解析 LLM 返回的报告 JSON。"""
        # 尝试提取 JSON
        json_str = raw_text.strip()
        if "```" in json_str:
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
            # 解析失败，返回基础信息
            user_msgs = [m for m in messages if m.get("role") == "user"]
            return LearningReport(
                session_id=session_id,
                total_interactions=len(messages),
                strengths=["报告解析失败"],
                suggestions=["请检查对话历史是否充足后重试"],
                engagement_score=min(len(user_msgs) * 10, 100),
            )

        topics = data.get("topics", [])
        strengths = data.get("strengths", [])
        weak_areas = data.get("weak_areas", [])
        suggestions = data.get("suggestions", [])
        engagement = data.get("engagement_score", 0)

        # 确保是列表
        if isinstance(topics, str):
            topics = [topics]
        if isinstance(strengths, str):
            strengths = [strengths]
        if isinstance(weak_areas, str):
            weak_areas = [weak_areas]
        if isinstance(suggestions, str):
            suggestions = [suggestions]

        return LearningReport(
            session_id=session_id,
            total_interactions=len(messages),
            topics=topics,
            strengths=strengths,
            weak_areas=weak_areas,
            suggestions=suggestions,
            engagement_score=float(engagement) if engagement else 0.0,
            generated_at="刚刚",
        )

    def generate_cross_session_summary(self, session_ids: List[str]) -> str:
        """
        跨会话综合分析（生成一段总结性描述）。

        Args:
            session_ids: 要分析的会话 ID 列表

        Returns:
            总结文本
        """
        all_messages = []
        for sid in session_ids:
            msgs = self.db.get_all_messages(sid)
            all_messages.extend(msgs)

        if not all_messages:
            return "没有足够的对话数据生成综合分析。"

        history_text = self._build_history_summary(all_messages[:200])  # 限制长度

        prompt = (
            f"请根据以下多轮对话历史，生成一段简短的学习情况综述：\n\n"
            f"{history_text}\n\n"
            f"请用 200 字以内的中文总结该学生的学习情况。"
        )

        try:
            response = _ollama_client.chat(
                model=config.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": 500, "temperature": 0.3},
            )
            return response["message"]["content"]
        except Exception as e:
            return f"生成综述失败：{str(e)}"
