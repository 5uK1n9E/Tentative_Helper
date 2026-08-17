"""
AI 智能助教助手 - Streamlit 主界面

四大功能模块：
1. 智能问答：基于课程知识库的 RAG 问答
2. 作业批改：文本/代码作业自动评分
3. 对话历史：查看和管理所有会话记录
4. 学习报告：基于对话历史生成个性化分析报告

侧边栏：知识库管理（上传文档、查看已导入文档）
"""

import os
import uuid
import traceback

import streamlit as st

# 确保项目根目录在 sys.path 中
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="AI 智能助教助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 懒加载模块（避免启动时阻塞）
# ============================================================
def _init_rag_engine():
    try:
        from core.rag_engine import RAGEngine
        return RAGEngine()
    except Exception as e:
        return None


def _init_chat_manager():
    try:
        from db.database import Database
        from core.chat_manager import ChatManager
        db = Database()
        return ChatManager(db=db)
    except Exception:
        return None


def _init_grader():
    try:
        from core.grader import Grader
        g = Grader()
        st.toast("✅ 批改引擎已就绪", icon="✅")
        return g
    except Exception as e:
        st.error(f"批改引擎初始化失败: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


def _init_report_generator():
    try:
        from db.database import Database
        from core.report_generator import ReportGenerator
        db = Database()
        rg = ReportGenerator(db=db)
        st.toast("✅ 报告生成器已就绪", icon="✅")
        return rg
    except Exception as e:
        st.error(f"报告生成器初始化失败: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


def get_rag_engine():
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = _init_rag_engine()
    return st.session_state.rag_engine


def get_chat_manager():
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = _init_chat_manager()
    return st.session_state.chat_manager


def get_grader():
    if "grader" not in st.session_state:
        st.session_state.grader = _init_grader()
    return st.session_state.grader


def get_report_generator():
    if "report_generator" not in st.session_state:
        st.session_state.report_generator = _init_report_generator()
    return st.session_state.report_generator


# 初始化会话 ID
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]


# ============================================================
# 侧边栏：知识库管理
# ============================================================
with st.sidebar:
    st.title("🤖 AI 智能助教")
    st.caption(f"当前会话: **{st.session_state.session_id}**")

    st.divider()

    # 知识库管理
    st.header("📚 知识库管理")

    uploaded_files = st.file_uploader(
        "上传课程资料（TXT/MD/PDF/DOCX）",
        accept_multiple_files=True,
        type=["txt", "md", "pdf", "docx"],
        help="支持纯文本、Markdown、PDF 和 Word 文档",
    )

    rag = get_rag_engine()
    if uploaded_files and rag:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(config.KNOWLEDGE_BASE_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(f"正在处理: {uploaded_file.name}..."):
                result = rag.ingest_document(file_path)

            if result["success"]:
                st.success(f"✅ {result['filename']} → {result['chunks_count']} 个片段")
            else:
                st.error(f"❌ {result['filename']}: {result['error']}")
    elif uploaded_files and not rag:
        st.error("知识库引擎初始化失败，请检查依赖是否安装完整")

    # 已导入文档列表
    if rag:
        docs = rag.list_documents()
        if docs:
            st.subheader("已导入文档")
            for doc in docs:
                st.caption(f"📄 {doc['filename']} ({doc['chunks_count']} 片段)")
            st.caption(f"总计: {rag.get_stats()['total_items']} 个片段")
        else:
            st.info("📭 暂无文档，请上传课程资料")
    else:
        st.info("📭 知识库未就绪")

    st.divider()

    # 会话管理
    st.header("💬 会话管理")
    cm = get_chat_manager()
    if cm:
        sessions = cm.get_all_sessions()
        if sessions:
            session_ids = [s["session_id"] for s in sessions]
            selected_session = st.selectbox(
                "选择会话",
                options=session_ids,
                index=0 if session_ids else None,
                key="session_selector",
            )
            if st.button("🗑️ 清空当前会话"):
                cm.delete_session(st.session_state.session_id)
                st.success("会话已清空")
                st.rerun()
        else:
            st.caption("暂无历史会话")
    else:
        st.caption("对话管理器未就绪")

    st.divider()

    # 系统信息
    st.header("ℹ️ 系统信息")
    st.caption(f"模型: {config.MODEL_NAME}")
    if rag:
        stats = rag.get_stats()
        st.caption(f"知识库片段: {stats['total_items']}")
        st.caption(f"已导入文档: {stats['document_count']}")
    else:
        st.caption("知识库: 未就绪")
    if cm:
        st.caption(f"对话会话: {cm.get_session_count()}")
    else:
        st.caption("对话管理: 未就绪")


# ============================================================
# 主内容区：功能标签页
# ============================================================
tab_qa, tab_grade, tab_history, tab_report = st.tabs([
    "💬 智能问答",
    "📝 作业批改",
    "📋 对话历史",
    "📊 学习报告",
])


# ============================================================
# Tab 1: 智能问答
# ============================================================
with tab_qa:
    st.markdown("### 💬 智能问答")
    st.caption("向 AI 助教提问课程相关问题，系统将基于知识库检索后作答。")

    # 初始化问答历史
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    # 输入框
    question = st.chat_input("请输入你的问题...")

    if question:
        cm = get_chat_manager()
        rag = get_rag_engine()

        if cm:
            cm.save_message(
                st.session_state.session_id, "user", question, source="qa"
            )
        st.session_state.qa_history.append({"role": "user", "content": question})

        # 检索知识库
        retrieved = []
        if rag:
            with st.spinner("🔍 正在检索知识库..."):
                try:
                    retrieved = rag.retrieve(question, top_k=config.TOP_K_RETRIEVE)
                except Exception as e:
                    st.warning(f"检索失败: {str(e)}，将使用直接对话模式")
                    retrieved = []

        if not retrieved:
            answer = "知识库尚未就绪或未找到相关内容。尝试直接提问："
            # 直接调用 Ollama
            with st.spinner("🤖 正在思考..."):
                try:
                    from ollama import Client as OllamaClient
                    _client = OllamaClient(host=config.OLLAMA_BASE_URL)
                    resp = _client.chat(
                        model=config.MODEL_NAME,
                        messages=[{
                            "role": "user",
                            "content": question,
                        }],
                        options={"num_predict": 1024, "temperature": 0.3},
                    )
                    answer = resp["message"]["content"]
                except Exception as e:
                    answer = f"⚠️ AI 服务暂时不可用: {str(e)}\n\n请确保已安装并运行 Ollama（`ollama serve`）且已拉取模型（`ollama pull qwen2.5:7b`）"
        else:
            # 构建 RAG 问答
            context_text = "\n\n".join(
                f"[{i+1}] ({r['source']}):\n{r['content']}"
                for i, r in enumerate(retrieved)
            )

            with st.spinner("🤖 正在思考..."):
                try:
                    from ollama import Client as OllamaClient
                    _client = OllamaClient(host=config.OLLAMA_BASE_URL)
                    resp = _client.chat(
                        model=config.MODEL_NAME,
                        messages=[{
                            "role": "user",
                            "content": (
                                f"你是 AI 课程助教。请根据以下资料回答问题。\n\n"
                                f"【资料】\n{context_text}\n\n"
                                f"【问题】{question}\n\n请简洁回答："
                            ),
                        }],
                        options={"num_predict": 1024, "temperature": 0.3},
                    )
                    answer = resp["message"]["content"]
                except Exception as e:
                    answer = f"⚠️ 生成回答失败: {str(e)}"

        # 保存助手回复
        if cm:
            cm.save_message(
                st.session_state.session_id, "assistant", answer, source="qa"
            )
        st.session_state.qa_history.append({"role": "assistant", "content": answer})

    # 显示对话历史
    for msg in st.session_state.qa_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ============================================================
# Tab 2: 作业批改
# ============================================================
with tab_grade:
    st.markdown("### 📝 作业批改")
    st.caption("提交作业内容，AI 将根据评分标准自动打分并提供详细反馈。")

    grader = get_grader()

    if grader:
        rubrics = grader.list_rubrics()
        if rubrics:
            rubric_options = {r["name"]: r["key"] for r in rubrics}
            selected_rubric_name = st.selectbox(
                "选择评分标准",
                options=list(rubric_options.keys()),
                index=0,
            )
            rubric_key = rubric_options[selected_rubric_name]

            # 显示该标准的详细信息
            rubric_info = next((r for r in rubrics if r["key"] == rubric_key), None)
            if rubric_info:
                with st.expander("📋 查看评分标准详情"):
                    st.markdown(f"**满分：** {rubric_info['max_score']} 分")
                    for criterion in rubric_info.get("criteria", []):
                        st.markdown(f"- **{criterion['name']}**（{criterion['weight']}分）：{criterion['description']}")
        else:
            rubric_key = "essay"
            selected_rubric_name = "作文/简答题"
    else:
        st.error("批改引擎未就绪，请检查依赖安装")
        rubric_key = "essay"
        selected_rubric_name = "作文/简答题"

    # 作业类型
    assignment_type = st.radio(
        "作业类型",
        options=["文本作业", "代码作业"],
        horizontal=True,
    )

    # 输入区域
    if assignment_type == "文本作业":
        submission = st.text_area(
            "请输入作业内容",
            height=200,
            placeholder="在此粘贴或输入你的作业内容...",
        )
        requirement = ""
    else:
        submission = st.text_area(
            "请输入代码",
            height=200,
            placeholder="# 在此粘贴你的代码...",
            key="code_input",
        )
        requirement = st.text_area(
            "题目要求（可选）",
            height=100,
            placeholder="描述题目的具体要求，例如：实现一个快速排序算法...",
            key="code_requirement",
        )

    if st.button("🚀 开始批改", type="primary", use_container_width=True):
        if not submission.strip():
            st.warning("⚠️ 请先输入作业内容或代码")
        elif not grader:
            st.error("批改引擎未就绪")
        else:
            with st.spinner("🤖 AI 正在批改中，请稍候..."):
                try:
                    result = grader.grade(
                        submission=submission,
                        rubric_key=rubric_key,
                        assignment_type="code" if assignment_type == "代码作业" else "text",
                        assignment_requirement=requirement,
                    )

                    # 显示结果
                    st.markdown("---")
                    st.markdown(result.to_markdown())

                    # 保存到对话历史
                    cm = get_chat_manager()
                    if cm:
                        cm.save_conversation(
                            st.session_state.session_id,
                            user_msg=f"【{selected_rubric_name}】\n{submission[:200]}",
                            assistant_msg=result.to_markdown(),
                            source="grade",
                        )

                except Exception as e:
                    st.error(f"❌ 批改失败: {str(e)}")
                    st.info("💡 请确保已安装 Ollama 并拉取了模型（`ollama pull qwen2.5:7b`）")
                    st.code(traceback.format_exc())


# ============================================================
# Tab 3: 对话历史
# ============================================================
with tab_history:
    st.markdown("### 📋 对话历史")
    st.caption("查看和管理所有历史对话记录。")

    cm = get_chat_manager()

    if cm:
        sessions = cm.get_all_sessions()

        if not sessions:
            st.info("📭 暂无对话记录，先去「智能问答」提个问题吧！")
        else:
            session_ids = [s["session_id"] for s in sessions]
            selected_sid = st.selectbox(
                "选择会话",
                options=session_ids,
                index=0,
            )

            if selected_sid:
                messages = cm.get_all_messages(selected_sid)

                # 统计信息
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总消息数", len(messages))
                with col2:
                    user_count = len([m for m in messages if m["role"] == "user"])
                    st.metric("提问次数", user_count)
                with col3:
                    source_stats = cm.get_source_stats(selected_sid)
                    st.metric("来源分布", ", ".join(f"{k}:{v}" for k, v in source_stats.items()))

                st.divider()

                # 按时间正序显示消息
                for msg in messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                        st.caption(f"来源: {msg.get('source', 'qa')} · {msg.get('timestamp', '')}")
    else:
        st.error("对话管理器未就绪")


# ============================================================
# Tab 4: 学习报告
# ============================================================
with tab_report:
    st.markdown("### 📊 学习报告")
    st.caption("基于对话历史，生成个性化学习分析报告。")

    cm = get_chat_manager()
    rg = get_report_generator()

    if cm:
        sessions = cm.get_all_sessions()

        if not sessions:
            st.info("📭 暂无对话数据，无法生成报告。请先进行一些问答或批改操作。")
        else:
            session_ids = [s["session_id"] for s in sessions]
            selected_sid = st.selectbox(
                "选择要分析的会话",
                options=session_ids,
                index=0,
            )

            if st.button("📊 生成学习报告", type="primary", use_container_width=True):
                with st.spinner("🤖 正在分析学习数据，请稍候..."):
                    if rg:
                        try:
                            report = rg.generate(selected_sid)

                            st.markdown("---")
                            st.markdown(report.to_markdown())

                            # 可视化参与度评分
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**参与度评分**")
                                st.progress(report.engagement_score / 100)
                                st.caption(f"{report.engagement_score:.0f} / 100")
                            with col2:
                                st.markdown("**交互统计**")
                                st.metric("总交互次数", report.total_interactions)

                        except Exception as e:
                            st.error(f"❌ 报告生成失败: {str(e)}")
                            st.code(traceback.format_exc())
                    else:
                        st.error("报告生成器未就绪")
    else:
        st.error("对话管理器未就绪")

    st.divider()

    # 跨会话综合分析
    st.markdown("### 🔀 跨会话综合分析")
    if cm and sessions:
        session_ids = [s["session_id"] for s in sessions]
        if len(session_ids) > 1:
            multi_select = st.multiselect(
                "选择要分析的会话（多选）",
                options=session_ids,
                default=session_ids[:min(3, len(session_ids))],
            )
            if multi_select and len(multi_select) > 1:
                if st.button("📈 生成综合分析"):
                    with st.spinner("正在分析..."):
                        if rg:
                            summary = rg.generate_cross_session_summary(multi_select)
                            st.markdown(summary)
                        else:
                            st.error("报告生成器未就绪")
        else:
            st.caption("需要至少两个会话才能进行跨会话分析")


# ============================================================
# 底部信息
# ============================================================
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #888; padding: 10px;'>
    AI 智能助教助手 · 小白的人工智能期末大作业 · 基于 Ollama + LangChain + Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
