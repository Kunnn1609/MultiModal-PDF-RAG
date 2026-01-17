# 🤖 智能 PDF 多模态问答系统 (Smart PDF RAG System)

> **基于 LangChain + Streamlit + Knowledge Graph 的通用文档智能分析助手**

![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)
![LangChain](https://img.shields.io/badge/Framework-LangChain-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![Status](https://img.shields.io/badge/Status-Active-success)

## 📖 项目简介

本项目是一个**工业级**的本地知识库问答系统。不同于传统的文本检索，本系统集成了**多模态解析**、**混合检索 (Hybrid Search)** 和**实时知识图谱**技术。

用户只需上传任意 PDF 文档（如技术报告、论文、合同），系统即可自动解析文本与图表，构建语义索引。在问答过程中，AI 不仅提供精准回答，还能**直接渲染 PDF 原文截图**，并生成思维图谱展示推理路径。

## ✨ 核心功能 (Key Features)

- **📚 多模态智能解析**：集成 **PaddleOCR**，精准提取 PDF 中的文本、表格及图片文字。
- **🔍 混合检索机制**：
  - **语义检索 (Vector Search)**：使用 ChromaDB + Text2Vec 理解问题深层含义。
  - **关键词检索 (BM25)**：精准匹配专有名词和术语。
  - **重排序 (Rerank)**：引入 BGE-Reranker 对召回结果进行二次精排，大幅提升准确率。
- **📸 原文“所见即所得”**：AI 回答时会自动定位页码，并直接在 UI 上渲染出 PDF **对应页面的高清截图**，拒绝“幻觉”。
- **🕸️ 动态知识图谱**：基于当前问答上下文，实时抽取实体关系并绘制可视化图谱。
- **📊 AI 质量审判官**：内置评估模块，对 AI 的回答进行“忠实度”、“相关性”打分。

---

## 📂 目录结构 (Directory Structure)

```text
PDF_RAG_Project/
├── app.py                  # 🚀 [启动入口] Streamlit 前端主程序
├── .env                    # 🔐 [配置文件] 存放 API Key (需手动创建，勿上传)
├── requirements.txt        # 📦 [依赖清单] 项目所需的 Python 库
├── .gitignore              # 🛑 [Git配置] 忽略数据和敏感文件
├── README.md               # 📄 项目说明文档
│
├── data/                   # 💾 [数据中心] (默认忽略，不上传)
│   ├── raw/                # 存放用户上传的原始 PDF 文件
│   └── vector_dbs/         # 存放生成的 Chroma 向量数据库和 BM25 索引
│
└── src/                    # 🧠 [核心源码]
    ├── llm/
    │   ├── rag_chain.py    # RAG 问答链路 (改写、检索、生成)
    │   ├── graph_agent.py  # 知识图谱抽取逻辑
    │   └── ...
    ├── parser/
    │   ├── smart_parser.py # PDF 解析与 OCR 处理核心
    │   └── ...
    ├── rag/
    │   ├── vector_storage.py # 向量库构建与存储逻辑
    │   └── reranker.py     # 重排序模型加载
    └── evaluation/
        └── evaluator.py    # AI 质量评估模块
```
---

## 🛠️ 安装与运行 (Installation & Setup)

请按照以下步骤，在本地环境中部署并运行本项目。

### 1. 环境准备 (Prerequisites)
建议使用 Anaconda 创建独立的虚拟环境，以避免依赖冲突。

```bash
# 创建名为 pdf_rag 的虚拟环境 (推荐 Python 3.9)
conda create -n pdf_rag python=3.9

# 激活环境
conda activate pdf_rag
```
## 🚀 快速启动 (Quick Start)

### 1. 获取代码
```bash
# 克隆仓库
git clone [https://github.com/你的用户名/你的仓库名.git](https://github.com/你的用户名/你的仓库名.git)

# 进入项目目录
cd PDF_RAG_Project
```

### 2. 安装依赖 (Install Dependencies)

本项目依赖多个深度学习和 Web 框架。为了确保所有功能（OCR、图谱、检索）正常运行，请按以下步骤操作：

```bash
# 1. (推荐) 使用清华源加速 pip 下载
pip config set global.index-url [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)

# 2. 安装项目基础依赖
pip install -r requirements.txt

# 3. ⚠️ 手动安装核心组件
# 某些库可能不在 requirements.txt 中，或者需要特定安装顺序，请务必执行：
pip install paddlepaddle paddleocr rank_bm25 pymupdf streamlit-agraph python-dotenv
```

### 3. 配置密钥 (API Key Configuration)

本项目使用阿里云通义千问 (Qwen) 作为推理核心，为了保护你的 Key 不被泄露，请使用环境变量方式配置。

1.  **创建文件**：在项目根目录（即 `app.py` 所在的文件夹）下，新建一个名为 `.env` 的文件。
    * *注意：文件名前面有个点，且没有后缀名（不要叫 .env.txt）。*

2.  **写入内容**：用记事本打开该文件，复制并填入你的 API Key：

    ```env
    DASHSCOPE_API_KEY=sk-你的通义千问API_KEY
    ```

3.  **保存**：保存并关闭文件。系统启动时会自动加载该密钥。

> **提示**：如果还没有 API Key，请前往 [阿里云百炼控制台](https://bailian.console.aliyun.com/) 免费申请。

---


```
## 💡 系统使用指南 (User Guide)

启动系统后（`streamlit run app.py`），请按照以下流程操作：
```
### 第一步：文档加载与解析
1.  在左侧 **“📂 文档管理”** 侧边栏，点击 **“Browse files”** 上传本地 PDF 文件。
2.  上传成功后，点击下方的 **“🚀 开始智能解析”** 按钮。
    * *系统将进行 OCR 识别、文本清洗及向量库构建，请耐心等待提示“✅ 入库完成”。*

### 第二步：智能问答
在主界面的对话框中输入关于文档的问题。
* **示例**：“这份合同的签署日期是哪天？” 或 “总结第二章的核心技术路线”。

### 第三步：多模态验证 (核心功能)
AI 回复生成后，你可以利用以下功能验证信息的真实性：
* **📸 原文截图回显**：点击回答下方的 **“📖 参考来源 & 原文截图”**。
    * 系统会自动展示 PDF 中与答案最相关的原始页面截图，所见即所得。
* **🕸️ 动态知识图谱**：观察下方的 **“🕸️ 当前思维图谱”**。
    * 查看 AI 提取的关键实体（节点）及其相互关系（连线）。

### 第四步：质量反馈
点击底部的 **“✨ 评分”** 按钮，调用 AI 审判官对当前回答的 **忠实度 (Faithfulness)** 和 **相关性 (Relevance)** 进行打分评测。