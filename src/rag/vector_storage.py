import os
import shutil
import time
import contextlib
from langchain.schema import Document 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

@contextlib.contextmanager
def temporary_chdir(path):
    """
    上下文管理器：临时切换工作目录
    用于解决 FAISS C++ 层无法处理中文绝对路径的 Bug
    """
    old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)

def build_vector_db(docs, db_name, embedding_model):
    """
    使用 FAISS 构建向量索引 (修复：正确读取 smart_parser 的元数据)
    """
    base_path = r"D:\workspace\finale_workspace\PDF_RAG_Project\data\vector_dbs"
    target_dir = os.path.join(base_path, db_name)
    
    # --- 1. 数据清洗与元数据提取 ---
    doc_objects = []
    
    for i, d in enumerate(docs):
        content = ""
        meta = {}
        
        if isinstance(d, dict):
            # 1. 提取内容
            content = d.get("page_content") or d.get("text") or d.get("content") or ""
            
            # 2. 提取元数据 (核心修复)
            # smart_parser 返回的是扁平字典，我们需要把非 content 的字段都放入 meta
            # 优先检查是否存在显式的 'page_number' (来自 parser)
            if "page_number" in d:
                meta["source_page"] = d["page_number"]
            if "method" in d:
                meta["method"] = d["method"]
                
            # 兼容其他格式：如果真有 metadata 键，也合并进来
            if "metadata" in d:
                meta.update(d["metadata"])
                
        else:
            # 兼容 Document 对象
            content = getattr(d, "page_content", "")
            meta = getattr(d, "metadata", {})

        # 3. 兜底逻辑：如果经过上述步骤还是没有页码，使用 i+1
        if "source_page" not in meta:
            meta["source_page"] = i + 1

        content = str(content)
        if not content or not content.strip():
            continue
            
        doc_objects.append(Document(page_content=content, metadata=meta))

    if not doc_objects:
        print("⚠️ [RAG] 警告：没有有效文档。")
        return None
    
    # --- 2. 清理旧数据 ---
    if os.path.exists(target_dir):
        try:
            shutil.rmtree(target_dir)
            print(f"🧹 旧索引已清理: {target_dir}")
        except Exception as e:
            print(f"⚠️ 清理旧文件失败: {e}")

    # --- 3. 切分文档 ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50
    )
    split_docs = text_splitter.split_documents(doc_objects)
    
    print(f"📄 文档切分完成: {len(doc_objects)} 页 -> {len(split_docs)} 个切片")
    
    # [Debug] 打印检查
    if len(split_docs) > 0:
        print(f"🐛 [Debug Check] 第一块元数据: {split_docs[0].metadata}")
        if len(split_docs) > 5:
            print(f"🐛 [Debug Check] 第五块元数据: {split_docs[5].metadata}")

    # --- 4. 构建并保存 FAISS 索引 ---
    try:
        print("🚀 正在构建 FAISS 内存索引...")
        vectorstore = FAISS.from_documents(
            documents=split_docs, 
            embedding=embedding_model
        )
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        print(f"💾 正在保存索引到: {target_dir}")
        with temporary_chdir(target_dir):
            vectorstore.save_local(".")
            
        print(f"✅ [RAG] FAISS 索引保存成功！")
        
    except Exception as e:
        print(f"❌ [RAG] 索引构建失败: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    return target_dir