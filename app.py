import streamlit as st
import os
import shutil
import time
import fitz  # PyMuPDF
from langchain_community.embeddings import DashScopeEmbeddings
from paddleocr import PaddleOCR
from dotenv import load_dotenv

# 引入后端模块
from src.parser.smart_parser import smart_extract
from src.rag.vector_storage import build_vector_db
from src.llm.rag_chain import get_answer_stream

st.set_page_config(page_title="智能文档专家 (Ultimate)", page_icon="⚡", layout="wide")
load_dotenv()

RAW_DATA_DIR = os.path.join("data", "raw")
DB_DATA_DIR = os.path.join("data", "vector_dbs")
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(DB_DATA_DIR, exist_ok=True)

if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0

@st.cache_resource
def load_ocr_engine():
    return PaddleOCR(use_angle_cls=True, lang="ch")

def render_pdf_page_as_image(pdf_path, human_page_num):
    if not os.path.exists(pdf_path): return None
    try:
        doc = fitz.open(pdf_path)
        try: page_index = int(human_page_num) - 1 
        except: page_index = 0
        if page_index < 0: page_index = 0
        if page_index >= len(doc): page_index = len(doc) - 1
        page = doc.load_page(page_index)
        
        # 🎨 【优化 1】降低渲染倍率：1.5倍足够清晰，且图片更小更轻量
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)) 
        return pix.tobytes()
    except: return None

def delete_project_completely(clean_filename):
    pdf_path = os.path.join(RAW_DATA_DIR, f"{clean_filename}.pdf")
    db_path = os.path.join(DB_DATA_DIR, clean_filename)
    if 'current_db' in st.session_state and st.session_state['current_db'] == db_path:
        del st.session_state['current_db']
    if 'last_selected' in st.session_state and st.session_state['last_selected'] == f"{clean_filename}.pdf":
        del st.session_state['last_selected']
    if os.path.exists(db_path):
        try: shutil.rmtree(db_path)
        except: return False
    if os.path.exists(pdf_path):
        try: os.remove(pdf_path)
        except: return False
    return True

# --- 🔥 核心修复：针对中文优化的评分算法 ---

def calculate_metrics(question, answer, source_docs):
    """
    使用字符级 (Character-level) Jaccard 相似度来评估中文质量
    """
    context_text = "".join([d.page_content for d in source_docs])
    ignore_chars = set(" ，。！？、\n\t*`")
    ans_chars = set(answer) - ignore_chars
    ctx_chars = set(context_text) - ignore_chars
    q_chars = set(question) - ignore_chars
    
    if not ans_chars: 
        return {"faithfulness": 0.0, "relevance": 0.0, "evidence": 0.0}

    overlap_chars = ans_chars.intersection(ctx_chars)
    faithfulness = len(overlap_chars) / len(ans_chars)

    if not q_chars:
        relevance = 0.0
    else:
        q_overlap = ans_chars.intersection(q_chars)
        relevance = min((len(q_overlap) / len(q_chars)) * 2.0, 1.0)

    evidence_score = min(len(source_docs) / 4, 1.0)

    return {
        "faithfulness": faithfulness,
        "relevance": relevance,
        "evidence": evidence_score
    }

def generate_expert_critique(metrics):
    critiques = []
    f = metrics['faithfulness']
    if f > 0.85: critiques.append("✅ **可信度极高**：回答严格基于原文。")
    elif f > 0.6: critiques.append("⚠️ **可信度一般**：包含部分总结性措辞。")
    else: critiques.append("🚫 **存在幻觉风险**：大量用词未在原文出现，请核对。")
    
    r = metrics['relevance']
    if r > 0.6: critiques.append("🎯 **切题精准**：紧扣问题核心。")
    elif r > 0.3: critiques.append("👌 **基本切题**：回答了主要方面。")
    else: critiques.append("🤔 **答非所问**：未包含问题关键词。")
    
    e = metrics['evidence']
    if e >= 0.75: critiques.append("📚 **引用丰富**：论证扎实。")
    else: critiques.append("🔍 **资料较少**：仅检索到少量片段。")
    
    return "\n\n".join(critiques)

# ================= 侧边栏 =================
with st.sidebar:
    st.header("📚 FAISS 书架")
    
    # 🛑 【优化 2】添加“中止生成”按钮 (新增内容)
    if st.button("⏹️ 中止/重置", type="primary"):
        st.rerun()
        
    st.divider()
    
    uploaded_file = st.file_uploader("➕ 上传", type=["pdf"], key=f"uploader_{st.session_state.uploader_key}")
    if uploaded_file:
        file_name = uploaded_file.name
        save_path = os.path.join(RAW_DATA_DIR, file_name)
        if not os.path.exists(save_path):
            with open(save_path, "wb") as f: f.write(uploaded_file.getbuffer())
            st.toast(f"✅ {file_name} 入库")
            st.session_state.uploader_key += 1
            time.sleep(0.5)
            st.rerun()
    
    st.divider()
    local_files = [f for f in os.listdir(RAW_DATA_DIR) if f.lower().endswith('.pdf')]
    if local_files:
        idx = 0
        if 'last_selected' in st.session_state and st.session_state['last_selected'] in local_files:
            idx = local_files.index(st.session_state['last_selected'])
        selected_file = st.selectbox("📂 选择文档", local_files, index=idx)
        st.session_state['last_selected'] = selected_file
        
        if selected_file:
            clean_name = os.path.splitext(selected_file)[0].strip()
            pdf_path = os.path.join(RAW_DATA_DIR, selected_file)
            db_path = os.path.join(DB_DATA_DIR, clean_name)
            st.session_state['current_pdf_path'] = pdf_path
            st.session_state['current_db'] = db_path
            
            if os.path.exists(os.path.join(db_path, "index.faiss")):
                st.success("✅ 已解析")
                if st.button("🗑️ 删除"):
                    if delete_project_completely(clean_name):
                        st.rerun()
            else:
                st.warning("⚠️ 未解析")
                if st.button("🚀 解析"):
                    with st.spinner("解析中..."):
                        try:
                            ocr = load_ocr_engine()
                            embed = DashScopeEmbeddings(model="text-embedding-v1")
                            raw = smart_extract(pdf_path, ocr)
                            build_vector_db(raw, clean_name, embed)
                            st.success("完成")
                            st.rerun()
                        except Exception as e: st.error(str(e))

# ================= 主界面 =================
st.title("⚡ PDF智能文档专家")

if 'messages' not in st.session_state: st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("提问..."):
    current_db = st.session_state.get('current_db')
    current_pdf = st.session_state.get('current_pdf_path')
    
    if not current_db or not os.path.exists(os.path.join(current_db, "index.faiss")):
        st.toast("❌ 请先解析文档")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        embed = DashScopeEmbeddings(model="text-embedding-v1")
        
        try:
            response_stream, source_docs = get_answer_stream(prompt, current_db, st.session_state.messages, embed)
            
            for chunk in response_stream:
                if chunk.status_code == 200:
                    content = chunk.output.choices[0].message.content
                    full_response += content
                    placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            if source_docs:
                unique_pages = sorted(list(set(
                    [doc.metadata.get('human_page_number', 1) for doc in source_docs]
                )))
                
                st.divider()
                st.markdown(f"**📚 引用来源 ({len(unique_pages)} 页)**")
                
                # 🎨 【优化 3】改为双列布局 (修改部分)
                cols = st.columns(2)
                
                for idx, page_num in enumerate(unique_pages):
                    # 将图片分配到左右两列
                    with cols[idx % 2]:
                        with st.expander(f"📄 第 {page_num} 页原文快照", expanded=True):
                            relevant_text = next((d.page_content for d in source_docs if d.metadata.get('human_page_number') == page_num), "...")
                            st.caption(f"相关内容摘录: ...{relevant_text[:100]}...")
                            if current_pdf and os.path.exists(current_pdf):
                                img_bytes = render_pdf_page_as_image(current_pdf, page_num)
                                # use_column_width=True 配合 columns(2) 会自动缩小图片
                                if img_bytes: st.image(img_bytes, use_column_width=True)
            
            # 计算指标
            scores = calculate_metrics(prompt, full_response, source_docs)
            
            # --- 仪表盘展示 ---
            st.divider()
            st.subheader("📊 质量评估")
            c1, c2, c3 = st.columns(3)
            c1.metric("🛡️ 忠实度", f"{scores['faithfulness']*100:.0f}%")
            c1.progress(scores['faithfulness'])
            c2.metric("🎯 相关性", f"{scores['relevance']*100:.0f}%")
            c2.progress(scores['relevance'])
            c3.metric("📚 引用数", len(source_docs))
            c3.progress(scores['evidence'])
            
            st.info(f"**🧑‍🏫 专家点评：**\n\n{generate_expert_critique(scores)}")
            
        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())