from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

import os
import pickle

# 1. 初始化中文 Embedding 模型 (推荐使用 BGE 或 m3e)
# 该模型将文字转为向量数字，是检索的基础
# db_path = r"D:\workspace\finale_workspace\PDF_RAG_Project\data\chroma_db"

def build_vector_db(full_content, file_name, embedding_model, base_db_path=r"D:\workspace\finale_workspace\PDF_RAG_Project\data\vector_dbs"):
    """
    为每个文件创建独立的向量库文件夹
    """
    # 1. 移除非法字符
    safe_name = file_name.replace(".pdf", "").replace(" ", "_")
    # 2. 确定最终存放路径
    save_path = os.path.join(base_db_path, safe_name)
    
    # 如果该文件的库已经存在，可以选择跳过或重新覆盖
    if os.path.exists(save_path):
        print(f"ℹ️ 文件 {file_name} 的知识库已存在，将直接复用。")
        # 如果你想强制覆盖，可以在这里用 shutil.rmtree(save_path)
        return save_path

    """
    将解析出的内容切片并存入向量库
    full_content: 之前 smart_extract 返回的页面字典列表
    """
    
    # 2. 配置切片器：控制块大小在 500 字左右，重叠 50 字
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )

    documents = []
    metadatas = []

    # 3. 遍历每一页，生成带页码元数据的切片
    for page in full_content:
        page_text = page['content']
        page_num = page['page_number']
        chunks = text_splitter.split_text(page_text)
        
        for chunk in chunks:
            documents.append(chunk)
            # 这里的 metadata 是后续解决幻觉、定位页码的唯一依据
            metadatas.append({
                "source_page": page_num,
                "extraction_method": page['method']
            })

    # 4. 创建并持久化向量库
    print(f"📦 正在构建向量库，当前共有 {len(documents)} 个知识切片...")
    vectordb = Chroma.from_texts(
        texts=documents,
        embedding=embedding_model,
        metadatas=metadatas,
        persist_directory=save_path
    )
    
    # 5. 【核心优化】构建并保存 BM25 检索器所需的数据
    # BM25 不像 Chroma 能自动持久化，我们需要手动保存文档列表
    print(f"🧬 正在生成关键词索引 (BM25)...")
    bm25_data = {
        "documents": documents,
        "metadatas": metadatas
    }
    with open(os.path.join(save_path, "bm25_data.pkl"), "wb") as f:
        pickle.dump(bm25_data, f)

    print(f"✅ 混合索引构建成功！路径: {save_path}")
    return vectordb

if __name__ == "__main__":
    # --- 全链路测试 (因为函数改了，这里测试代码也要改) ---
    import sys
    from langchain_community.embeddings import HuggingFaceEmbeddings # 仅测试时导入
    
    # 添加 src 目录到路径
    src_dir = os.path.join(os.path.dirname(__file__), '..')
    src_dir = os.path.abspath(src_dir)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    from parser.smart_parser import smart_extract
    from paddleocr import PaddleOCR
    
    # 1. 初始化模型 (这是模拟 app.py 的行为)
    print("⏳ 测试模式：正在初始化 Embedding 模型...")
    test_embedding_model = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    
    # 2. 模拟解析
    engine = PaddleOCR(lang="ch", use_angle_cls=True)
    # 替换为你本地真实存在的 PDF 路径
    test_pdf = r"D:\workspace\finale_workspace\PDF_RAG_Project\data\raw\test.pdf" 
    
    if os.path.exists(test_pdf):
        print("🔍 开始解析 PDF...")
        pages_data = smart_extract(test_pdf, engine)
        
        # 3. 存入数据库 (传入模型)
        print("💾 开始建库...")
        # 注意：这里传入了 test_embedding_model
        db = build_vector_db(pages_data, "test.pdf", test_embedding_model)
        
        if db:
            # 4. 验证检索功能
            query = "测试提问"
            print(f"\n🔍 测试检索提问: {query}")
            results = db.similarity_search(query, k=2)
            for doc in results:
                print(f"[P{doc.metadata['source_page']}] {doc.page_content[:50]}...")
    else:
        print(f"❌ 测试文件不存在: {test_pdf}")