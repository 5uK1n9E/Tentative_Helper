"""
对话管理器单元测试
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import Database
from core.chat_manager import ChatManager


def test_chat_manager_save_and_retrieve():
    """测试保存和检索对话消息。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = Database(db_path)
        manager = ChatManager(db=db)

        # 保存消息
        manager.save_message("test_session", "user", "你好，请问什么是机器学习？")
        manager.save_message("test_session", "assistant", "机器学习是AI的核心分支...")

        # 检索
        history = manager.get_history("test_session")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        print("✅ 保存和检索测试通过")


def test_save_conversation():
    """测试批量保存一轮对话。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = Database(db_path)
        manager = ChatManager(db=db)

        manager.save_conversation(
            "conv_test",
            "用户问题",
            "助手回答",
            source="qa",
        )

        history = manager.get_history("conv_test")
        assert len(history) == 2
        assert history[0]["source"] == "qa"
        print("✅ 批量保存对话测试通过")


def test_delete_session():
    """测试删除会话。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = Database(db_path)
        manager = ChatManager(db=db)

        manager.save_message("del_test", "user", "test")
        manager.delete_session("del_test")

        history = manager.get_history("del_test")
        assert len(history) == 0
        print("✅ 删除会话测试通过")


def test_get_all_sessions():
    """测试列出所有会话。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = Database(db_path)
        manager = ChatManager(db=db)

        manager.save_message("sess_a", "user", "question 1")
        manager.save_message("sess_b", "user", "question 2")

        sessions = manager.get_all_sessions()
        session_ids = [s["session_id"] for s in sessions]
        assert "sess_a" in session_ids
        assert "sess_b" in session_ids
        print(f"✅ 列出所有会话测试通过，共 {len(sessions)} 个会话")


if __name__ == "__main__":
    test_chat_manager_save_and_retrieve()
    test_save_conversation()
    test_delete_session()
    test_get_all_sessions()
    print("\n🎉 所有对话管理测试通过！")
