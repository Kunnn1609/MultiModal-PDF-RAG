import dashscope
import os
import contextlib
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

# --- 0. 辅助工具：中文路径补丁 ---
@contextlib.contextmanager
def temporary_chdir(path):
    old_cwd = os.getcwd()
    os.chdir(path)
    try: yield
    finally: os.chdir(old_cwd)

# --- 1. Rerank (重排序) ---
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

    # Step 3: 检索 (海量召回)
    # 召回 20 条，确保覆盖全文主要内容
    retrieved_docs = vectorstore.similarity_search(search_query, k=20)

    # Step 4: Rerank (精选)
    # 选出最相关的 10 条给大模型
    final_docs = rerank_documents(search_query, retrieved_docs, top_k=10)
    
    # Step 5: 构建上下文 (🔥 关键：清洗页码，让 Prompt 看得懂)
    # 我们先对文档进行排序，让页码从小到大，符合阅读逻辑
    for doc in final_docs:
        raw_page = doc.metadata.get('source_page') or doc.metadata.get('page_number') or 1
        try:
            val = int(raw_page)
            # 如果索引是0，变成1；如果是1，保持1。
            # 这里的逻辑取决于你的 Parser 存的是 0-based 还是 1-based。
            # 假设之前出现过偏差，这里我们统一确保最小是 1。
            doc.metadata['human_page_number'] = val if val > 0 else 1
        except:
            doc.metadata['human_page_number'] = 1

    # 按页码排序
    final_docs.sort(key=lambda x: x.metadata['human_page_number'])

    context_list = []
    for doc in final_docs:
        p = doc.metadata['human_page_number']
        # 这里的格式要非常清晰，让 AI 知道这一段话属于哪一页
        context_list.append(f"【第 {p} 页内容】:\n{doc.page_content}")
    
    context_str = "\n\n".join(context_list) if context_list else "未找到相关文档。"

    # Step 6: Prompt (🔥 核心修改：强制引用格式)
    system_prompt = f"""你是一个极其严谨的文档分析员。你必须完全基于下方的【参考资料】回答用户问题。

### ⚠️ 核心原则 (必须遵守)：
1.  **零外部知识**：你的大脑里只有下方的【参考资料】，忘掉你训练过的其他知识。如果资料里没提到的内容，直接说“资料未提及”。
2.  **强制引用格式**：你的回答中，**每一句**事实陈述、数据引用或观点总结，都必须在该句结尾加上来源，格式严格为：`(来自第x页)`。
    * ❌ 错误：根据文档，营收增长了。
    * ✅ 正确：2023年营收增长了20%(来自第5页)。
    * ✅ 正确：作者认为时间管理是骗局(来自第1页)，真正的关键是注意力管理(来自第2页)。
3.  **页码对应**：参考资料中标记为【第 x 页内容】，你的引用就必须写 `(来自第x页)`，不要自己加减数字。

### 参考资料：
{context_str}
"""

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': query}
    ]

    responses = dashscope.Generation.call(
        model='qwen-turbo', # 或者 qwen-plus 效果更好
        messages=messages,
        result_format='message',
        stream=True,
        incremental_output=True
    )
    
    return responses, final_docs