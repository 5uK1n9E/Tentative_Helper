"""
对话管理器 - 负责多轮对话的记忆管理和消息持久化。

使用 SQLite 存储所有对话记录，支持：
- 按会话 ID 保存/查询消息
- 会话列表管理
- 消息源分类（问答/批改/报告）
"""

from db.database import Database


class ChatManager:
    """对话管理器，封装消息的保存和检索。"""

    def __init__(self, db: Database = None):
        """
        Args:
            db: Database 实例，如果不提供则自动创建
        """
        self.db = db or Database()

    def save_message(self, session_id: str, role: str, content: str,
                     source: str = "qa") -> int:
        """
        保存一条对话消息。

        Args:
            session_id: 会话 ID
            role: 角色（user / assistant / system）
            content: 消息内容
            source: 来源类型（qa / grade / report）

        Returns:
            新记录的 id
        """
        # 确保会话已创建
        self.db.create_session(session_id)
        return self.db.save_message(session_id, role, content, source)

    def save_conversation(self, session_id: str, user_msg: str,
                          assistant_msg: str, source: str = "qa"):
        """
        保存一轮完整的对话（用户提问 + 助手回答）。

        Args:
            session_id: 会话 ID
            user_msg: 用户消息
            assistant_msg: 助手回复
            source: 来源类型
        """
        self.db.create_session(session_id)
        self.db.save_messages_batch(session_id, [
            ("user", user_msg, source),
            ("assistant", assistant_msg, source),
        ])

    def get_history(self, session_id: str, last_n: int = 20) -> list:
        """
        获取指定会话的最近 N 条消息。

        Args:
            session_id: 会话 ID
            last_n: 消息数量上限

        Returns:
            消息列表，每项为 {role, content, source, timestamp}
        """
        return self.db.get_history(session_id, last_n)

    def get_all_messages(self, session_id: str) -> list:
        """获取指定会话的全部消息。"""
        return self.db.get_all_messages(session_id)

    def get_all_sessions(self) -> list:
        """
        列出所有会话。

        Returns:
            会话列表，每项为 {session_id, last_active}
        """
        return self.db.get_all_sessions()

    def get_session_count(self) -> int:
        """获取会话总数。"""
        return self.db.get_session_count()

    def get_source_stats(self, session_id: str) -> dict:
        """获取指定会话的来源统计。"""
        return self.db.get_source_stats(session_id)

    def delete_session(self, session_id: str) -> bool:
        """
        删除指定会话及其所有消息。

        Returns:
            是否成功删除
        """
        self.db.delete_session(session_id)
        return True

    def clear_all(self):
        """清空所有对话数据。"""
        self.db.clear_all()
