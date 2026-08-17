# AI 智能助教助手

> 小白的人工智能 · 期末大作业 · AI 赋能教学环节

## 📖 项目简介

本项目是一个基于本地大语言模型的 **AI 智能助教系统**，集成了智能问答、作业批改、对话历史管理和学习报告生成四大核心功能。系统完全在本地运行，无需任何云端 API Key，保护用户隐私的同时降低了使用门槛。

## ✨ 核心功能

| 功能 | 说明 |
|------|------|
| 💬 **智能问答** | 基于 RAG（检索增强生成）的课程问答，AI 助教根据知识库内容精准作答 |
| 📝 **作业批改** | 支持文本作业和代码作业的自动评分，多维度打分 + 详细反馈 |
| 📋 **对话历史** | 持久化存储所有对话记录，支持按会话查看和管理 |
| 📊 **学习报告** | 基于对话历史自动生成个性化学习分析报告，包含话题分析、强弱项评估和学习建议 |

## 🛠️ 技术栈

- **前端框架**：Streamlit（交互式 Web UI）
- **LLM 引擎**：Ollama + qwen2.5:7b（本地运行，中文优化）
- **嵌入模型**：sentence-transformers / all-MiniLM-L6-v2（离线向量化）
- **向量数据库**：ChromaDB（本地持久化存储）
- **关系型数据库**：SQLite（对话记录持久化）
- **AI 框架**：LangChain（Prompt 管理与 Chain 编排）

## 📦 安装与运行

### 前置条件

1. **Python 3.10+**
2. **Ollama**：[下载地址](https://ollama.com)

### 快速开始

```bash
# 1. 进入项目目录
cd "下载目录\AI Generator"

# 2. 安装依赖
pip install -r requirements.txt

# 3. 拉取模型（首次运行）
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# 4. 启动 Ollama 服务
ollama serve

# 5. 运行应用
streamlit run app.py
```

浏览器将自动打开 `http://localhost:8501`。

## 📁 项目结构

```
AI Generator/
├── app.py                      # Streamlit 主入口
├── config.py                   # 全局配置
├── requirements.txt            # Python 依赖
├── rubrics.json               # 评分标准配置
├── README.md                  # 项目说明
│
├── core/                      # 核心业务逻辑
│   ├── rag_engine.py          # RAG 知识库引擎
│   ├── chat_manager.py        # 对话管理器
│   ├── grader.py              # 作业批改引擎
│   └── report_generator.py    # 学习报告生成器
│
├── llm/                       # LLM 层
│   ├── model_client.py        # Ollama 客户端
│   ├── prompt_templates.py    # Prompt 模板
│   └── chain_factory.py       # Chain 工厂
│
├── db/                        # 数据库层
│   ├── schema.sql             # 建表脚本
│   └── database.py            # 数据库操作类
│
├── knowledge_base/
│   └── documents/             # 课程资料目录
│       └── sample_syllabus.txt
│
├── tests/                     # 测试文件
│   ├── test_rag.py
│   ├── test_grader.py
│   └── test_chat_manager.py
│
└── assets/
    └── styles.css             # 自定义样式
```

## 🎯 使用指南

### 智能问答
1. 在左侧边栏上传课程资料（PDF/TXT/DOCX/MD）
2. 切换到「智能问答」标签页
3. 输入问题，AI 将基于知识库内容作答
4. 支持多轮对话，系统自动维护上下文

### 作业批改
1. 切换到「作业批改」标签页
2. 选择评分标准和作业类型（文本/代码）
3. 输入或粘贴作业内容
4. 点击「开始批改」，查看评分和详细反馈

### 学习报告
1. 完成一定数量的问答或批改后
2. 切换到「学习报告」标签页
3. 选择要分析的会话
4. 点击「生成学习报告」，查看分析报告

## ⚙️ 配置说明

所有配置集中在 `config.py` 中，主要参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| MODEL_NAME | qwen2.5:7b | Ollama 模型名称 |
| EMBEDDING_MODEL | all-MiniLM-L6-v2 | 嵌入模型名称 |
| CHUNK_SIZE | 500 | 文档分块大小 |
| TOP_K_RETRIEVE | 5 | 检索返回片段数 |
| OLLAMA_BASE_URL | http://localhost:11434 | Ollama 服务地址 |

可通过环境变量覆盖默认配置：
```bash
export MODEL_NAME="llama3.2"
export OLLAMA_BASE_URL="http://127.0.0.1:11434"
```

## 📝 评分标准

系统内置三种评分标准（可在 `rubrics.json` 中自定义）：

- **作文/简答题**：内容准确性(5) + 逻辑结构(3) + 语言表达(2) = 10分
- **编程作业**：功能正确性(5) + 代码规范(3) + 算法效率(2) = 10分
- **项目作业**：功能完整性(4) + 技术深度(3) + 创新性与文档(3) = 10分

## 🔒 隐私说明

本系统所有数据处理均在本地完成：
- LLM 推理通过本地 Ollama 服务
- 知识库向量存储在本地 ChromaDB
- 对话记录存储在本地 SQLite
- **无任何数据上传至云端**

## 📄 许可证

本项目仅供教学用途。

## 👥 作者

小白的人工智能课程期末大作业
