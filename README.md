# 🤖 智能 PDF 多模态问答系统 (Smart PDF RAG System)

> 基于 LangChain + Streamlit + Knowledge Graph 的通用文档智能分析助手。

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![LangChain](https://img.shields.io/badge/Framework-LangChain-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

## 📖 项目简介
本项目是一个**工业级**的本地知识库问答系统。不同于传统的文本检索，本系统集成了**多模态解析**、**混合检索 (Hybrid Search)** 和**实时知识图谱**技术。

用户只需上传任意 PDF 文档，系统即可自动解析文本与图表，构建语义索引。在问答过程中，不仅提供精准的文本回答，还能**直接渲染 PDF 原文截图**，并生成思维图谱展示实体关系。

## ✨ 核心功能 (Key Features)

- **📚 多模态智能解析**：基于 PaddleOCR，支持对 PDF 中的文本、表格、图片文字进行深度提取。
- **🔍 混合检索机制**：
  - **语义检索 (Vector Search)**：使用 ChromaDB + Text2Vec 理解问题深层含义。
  - **关键词检索 (BM25)**：精准匹配专有名词和术语。
  - **重排序 (Rerank)**：引入 BGE-Reranker 对召回结果进行二次精排，大幅提升准确率。
- **🕸️ 动态知识图谱**：根据当前问答上下文，实时抽取实体关系并绘制可视化图谱 (Streamlit-Agraph)。
- **📸 原文“所见即所得”**：AI 回答时会自动定位页码，并直接在 UI 上渲染出 PDF **对应页面的截图**，拒绝幻觉。
- **📊 AI 质量审判官**：内置评估模块，对 AI 的回答进行“忠实度”、“相关性”打分。

## 🛠️ 技术栈

- **前端 UI**: Streamlit, Streamlit-Agraph
- **大模型 API**: 通义千问 (Qwen-Turbo)
- **RAG 框架**: LangChain, LangChain-Community
- **向量数据库**: Chroma
- **OCR 引擎**: PaddleOCR
- **PDF 处理**: PyMuPDF (Fitz)

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone [https://github.com/你的用户名/你的仓库名.git](https://github.com/你的用户名/你的仓库名.git)
cd PDF_RAG_Project