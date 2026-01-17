import sys
import os

# 1. 环境与路径处理
os.environ["PYTHONNOUSERSITE"] = "1"
sys.path = [p for p in sys.path if "AppData" not in p]
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
# 必须先安装: pip install pymupdf
import fitz  # PyMuPDF
from PIL import Image

st.set_page_config(page_title="智能PDF多模态问答系统", page_icon="📚", layout="wide")

import shutil
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.parser.smart_parser import smart_extract, PaddleOCR
from src.rag.vector_storage import build_vector_db
from src.llm.rag_chain import get_answer_stream
from src.evaluation.evaluator import evaluate_response
from src.llm.graph_agent import extract_triplets_from_text, build_graph_config
from src.rag.reranker import get_reranker
from streamlit_agraph import agraph

# 预加载模型
@st.cache_resource
def load_models():
    print("⏳ [系统] 正在初始化 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    try:
        get_reranker()
    except:
        pass
    return embeddings

with st.spinner("正在启动智能分析引擎..."):
    global_embed_model = load_models()

# 侧边栏逻辑
with st.sidebar:
    st.header("📂 文档管理")
    with st.expander("🛠️ 系统诊断"):
        st.write(f"**Python:** {sys.executable}")

    st.divider()
    DB_BASE_PATH = r"D:\workspace\finale_workspace\PDF_RAG_Project\data\vector_dbs"
    RAW_DATA_PATH = r"D:\workspace\finale_workspace\PDF_RAG_Project\data\raw"
    os.makedirs(DB_BASE_PATH, exist_ok=True)
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    
    uploaded_file = st.file_uploader("上传 PDF 文档", type="pdf")

    @st.cache_resource
    def init_ocr():
        return PaddleOCR(lang="ch", use_angle_cls=True)
    ocr_engine = init_ocr()
    
    if uploaded_file:
        # --- 核心修改：强制标准化文件名 ---
        # 1. 把文件名里的空格强制换成下划线，防止路径匹配错误
        clean_filename = uploaded_file.name.replace(" ", "_")
        
        # 2. 确定 PDF 存储路径 (存下去的就是标准化的名字)
        file_path = os.path.join(RAW_DATA_PATH, clean_filename)
        
        # 3. 确定数据库名称 (去掉 .pdf 后缀)
        # 这样 PDF 文件名 = "My_File.pdf"，数据库名 = "My_File"，完全对应
        db_name = clean_filename.replace(".pdf", "")
        target_db_path = os.path.join(DB_BASE_PATH, db_name)

        # 保存文件
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("🚀 开始智能解析"):
            with st.status("正在处理文档...", expanded=True) as status:
                st.write("🔍 OCR 文本/图表识别...")
                pages_data = smart_extract(file_path, ocr_engine)
                st.write("🧠 建立语义索引...")
                # 注意：这里传入 clean_filename，确保后续逻辑一致
                build_vector_db(pages_data, clean_filename, embedding_model=global_embed_model)
                status.update(label="✅ 入库完成", state="complete")
            st.rerun()

    st.divider()
    existing_dbs = [d for d in os.listdir(DB_BASE_PATH) if os.path.isdir(os.path.join(DB_BASE_PATH, d))]
    
    # 全局变量：当前选中的知识库名称
    selected_db_name = None 
    
    if existing_dbs:
        selected_db_name = st.selectbox("选择知识库：", existing_dbs)
        current_db_path = os.path.join(DB_BASE_PATH, selected_db_name)
        if st.button("🗑️ 删除"):
            shutil.rmtree(current_db_path)
            st.rerun()
    else:
        st.warning("暂无知识库")
        current_db_path = None

# 主界面
st.title("🤖 PDF 智能问答系统")
st.caption("🚀 支持多模态解析 · 混合检索 · 原文截图 · 知识图谱")
st.divider()

if "messages" not in st.session_state: st.session_state.messages = []
if "latest_qa_pair" not in st.session_state: st.session_state.latest_qa_pair = None

# --- 核心：渲染历史消息 (含 PDF 截图) ---
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant":
            # 1. 来源文档 + PDF 截图
            if "source_docs" in msg and msg["source_docs"]:
                with st.expander("📖 参考来源 & 原文截图"):
                    docs = msg["source_docs"]
                    # 使用 Tabs 切换不同来源
                    tabs = st.tabs([f"📄 P{d.metadata.get('source_page', '?')}" for d in docs[:3]])
                    
                    for idx, tab in enumerate(tabs):
                        with tab:
                            doc = docs[idx]
                            c1, c2 = st.columns([1, 1])
                            
                            # 左侧：文字内容
                            with c1:
                                st.caption("🔍 提取文本")
                                st.info(f"{doc.page_content}...")
                            
                            # 右侧：PDF 截图 (绝活功能)
                            with c2:
                                try:
                                    if selected_db_name:
                                        # --- 核心修改：直接拼接，无需猜测 ---
                                        # 因为我们在上传时已经强制统一了命名规则：
                                        # 数据库名 "My_File" -> 对应的 PDF 一定是 "My_File.pdf"
                                        pdf_name = selected_db_name + ".pdf"
                                        pdf_file_path = os.path.join(RAW_DATA_PATH, pdf_name)
                                        
                                        if os.path.exists(pdf_file_path):
                                            # 注意：source_page 有时候可能是 string，安全转 int
                                            page_num = int(doc.metadata.get('source_page', 1)) - 1
                                            
                                            with fitz.open(pdf_file_path) as pdf:
                                                # 安全检查页码范围
                                                if 0 <= page_num < len(pdf):
                                                    page = pdf[page_num]
                                                    # 缩放系数 2 表示 2 倍清晰度
                                                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                                                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                                    st.image(img, caption=f"📸 原文 P{page_num+1} 截图", use_column_width=True)
                                                else:
                                                    st.warning(f"页码 {page_num+1} 超出文档范围")
                                        else:
                                            # 如果找不到，打印一下路径方便调试
                                            st.warning(f"未找到源文件: {pdf_name}")
                                            st.caption(f"请确认 {RAW_DATA_PATH} 目录下是否存在该文件")
                                except Exception as e:
                                    st.error(f"截图加载失败: {e}")

# 输入处理
if prompt := st.chat_input("请输入关于文档的问题..."):
    if not current_db_path:
        st.error("请先上传或选择一个文档！")
        st.stop()

    # 用户消息上屏
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 助手回复
    with st.chat_message("assistant"):
        msg_placeholder = st.empty()
        full_response = ""
        
        responses, source_docs = get_answer_stream(
            prompt, 
            current_db_path, 
            st.session_state.messages,
            embedding_model=global_embed_model
        )
        
        for response in responses:
            if response.status_code == 200:
                chunk = response.output.choices[0].message.content
                full_response += chunk
                msg_placeholder.markdown(full_response + "▌")
        
        msg_placeholder.markdown(full_response)

        # 构建上下文
        context_str = "\n\n".join([f"[P{d.metadata.get('source_page')}] {d.page_content}" for d in source_docs])
        
        # 生成图谱
        current_graph = None
        with st.spinner("正在分析实体关系..."):
            triplets = extract_triplets_from_text(context_str)
            if triplets:
                nodes, edges, config = build_graph_config(triplets)
                current_graph = {"nodes": nodes, "edges": edges, "config": config}
            else:
                current_graph = "empty"

        # 打包存入历史
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response,
            "source_docs": source_docs,
            "graph_data": current_graph
        })

        # 保存最新数据用于下方展示
        st.session_state.latest_qa_pair = {
            "query": prompt,
            "context": context_str,
            "response": full_response,
            "graph": current_graph
        }
        
        st.rerun()

# --- 底部功能区 (只针对最新一条) ---
if st.session_state.latest_qa_pair:
    latest_data = st.session_state.latest_qa_pair
    
    # 1. 知识图谱 (只显示最新的，避免 key 报错)
    graph = latest_data.get("graph")
    if graph and graph != "empty":
        st.divider()
        st.subheader("🕸️ 当前思维图谱")
        agraph(nodes=graph["nodes"], edges=graph["edges"], config=graph["config"])

    # 2. 质量评估
    st.divider()
    with st.expander("📊 质量评估 (针对最新问答)"):
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("✨ 评分"):
                with st.spinner("评估中..."):
                    raw = evaluate_response(latest_data["query"], latest_data["context"], latest_data["response"])
                    try:
                        res = json.loads(raw.replace("```json", "").replace("```", "").strip())
                        st.info(f"忠实度: {res.get('faithfulness')}/10 | 相关性: {res.get('relevance')}/10")
                        st.caption(res.get('reason'))
                    except:
                        st.write(raw)