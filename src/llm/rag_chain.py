import dashscope
import os
import contextlib
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

# --- 0. 辅助工具 ---
@contextlib.contextmanager
def temporary_chdir(path):
    old_cwd = os.getcwd()
    os.chdir(path)
    try: yield
    finally: os.chdir(old_cwd)

# --- 1. Rerank ---
try:
    from src.rag.reranker import get_reranker
    def rerank_documents(query, docs, top_k=3):
        if not docs: return []
        reranker = get_reranker()
        pairs = [[query, d.page_content] for d in docs]
        scores = reranker.compute_score(pairs)
        if isinstance(scores, float): scores = [scores]
        doc_score_pairs = list(zip(docs, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in doc_score_pairs[:top_k]]
except ImportError:
    def rerank_documents(query, docs, top_k=3):
        return docs[:top_k]

# --- 2. API 配置 ---
load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# --- 3. 查询改写 ---
def rewrite_query(user_query, chat_history):
    if not chat_history: return user_query
    recent = chat_history[-2:]
    history_text = "\n".join([f"{'用户' if m['role']=='user' else '助手'}: {m['content']}" for m in recent])
    prompt = f"任务：改写提问，补全指代词。\n历史：{history_text}\n提问：{user_query}\n结果："
    try:
        # 改写不需要太严谨，temperature 保持默认即可
        res = dashscope.Generation.call(model='qwen-turbo', messages=[{'role':'user','content':prompt}], result_format='message')
        if res.status_code == 200: return res.output.choices[0].message.content.strip()
    except: pass
    return user_query

# --- 4. 核心主流程 ---
def get_answer_stream(query, db_path, chat_history=[], embedding_model=None):
    if embedding_model is None: raise ValueError("需要 embedding_model")
    if not os.path.exists(db_path): raise FileNotFoundError(f"找不到索引: {db_path}")

    # Step 1: 改写
    search_query = rewrite_query(query, chat_history)
    
    # Step 2: 加载 FAISS
    try:
        with temporary_chdir(db_path):
            vectorstore = FAISS.load_local(".", embedding_model, allow_dangerous_deserialization=True)
    except:
        vectorstore = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)

    # Step 3: 检索
    retrieved_docs = vectorstore.similarity_search(search_query, k=20)

    # Step 4: Rerank
    final_docs = rerank_documents(search_query, retrieved_docs, top_k=10)
    
    # Step 5: 构建上下文
    for doc in final_docs:
        raw_page = doc.metadata.get('source_page') or doc.metadata.get('page_number') or 1
        try:
            val = int(raw_page)
            doc.metadata['human_page_number'] = val if val > 0 else 1
        except:
            doc.metadata['human_page_number'] = 1

    final_docs.sort(key=lambda x: x.metadata['human_page_number'])

    context_list = []
    for doc in final_docs:
        p = doc.metadata['human_page_number']
        context_list.append(f"【第 {p} 页内容】:\n{doc.page_content}")
    
    context_str = "\n\n".join(context_list) if context_list else "未找到相关文档。"

    # Step 6: Prompt (🔥 优化重点：结构化思维链 Prompt)
    system_prompt = f"""你是一个专业的深度阅读助手。请基于【参考资料】回答问题。

### ⚠️ 回答规范 (必须严格遵守)：
1.  **结构化输出**：请务必使用 **Markdown 列表 (Bullet Points)** 的形式来组织答案，不要写成一大段长文。
2.  **细节优先**：不要只写宏观概念，要提取具体的**方法论、步骤、核心观点**。
    * ❌ 差的回答：文章介绍了时间管理的方法。
    * ✅ 好的回答：文章提出了 "5 AM Project"，建议利用早上5点的时间做最重要的事 (来自第2页)。
3.  **强制引用**：每一条要点后必须标注 `(来自第x页)`。
4.  **零外部知识**：严禁编造资料中没有的内容。

### 参考资料：
{context_str}
"""

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': query}
    ]

    # 🔥 核心修改：添加 temperature 参数
    responses = dashscope.Generation.call(
        model='qwen-turbo',
        messages=messages,
        result_format='message',
        stream=True,
        incremental_output=True,
        temperature=0.01,  # 👈 关键！设为极低值，接近 0
        top_p=0.8          # 辅助参数，限制过度发散
    )
    
    return responses, final_docs