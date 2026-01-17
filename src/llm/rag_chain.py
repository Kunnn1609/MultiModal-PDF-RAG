import dashscope
import os
from dotenv import load_dotenv
import pickle
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

# --- 尝试导入 Rerank 模块 ---
# 这是一个容错设计：如果 src.rag.reranker 没写好，系统会自动回退到不重排序模式，防止报错
try:
    from src.rag.reranker import get_reranker
    
    def rerank_documents(query, docs, top_k=3):
        """
        使用 BGE-Reranker 对文档进行精细化排序
        """
        if not docs:
            return []
        
        # 获取单例模型
        reranker = get_reranker()
        
        # 构造模型需要的输入对: [[Query, Doc1], [Query, Doc2]...]
        pairs = [[query, d.page_content] for d in docs]
        
        # 计算得分
        scores = reranker.compute_score(pairs)
        
        # 兼容性处理：如果只有一个文档，scores 可能是一个 float
        if isinstance(scores, float):
            scores = [scores]
            
        # 打包 (Doc, Score) 并按分数降序排列
        doc_score_pairs = list(zip(docs, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前 Top-K 个文档
        return [doc for doc, score in doc_score_pairs[:top_k]]

except ImportError:
    print("⚠️ 未找到 Rerank 模块 (src.rag.reranker)，将跳过重排序步骤。")
    def rerank_documents(query, docs, top_k=3):
        return docs[:top_k]


# 配置 API KEY
# 加载 .env 文件里的环境变量
load_dotenv()
# 请用户自己填入 Key
api_key = os.getenv("DASHSCOPE_API_KEY")
if api_key is None:
    raise ValueError("⚠️ 未找到 DASHSCOPE_API_KEY，请检查 .env 文件！")

dashscope.api_key = api_key
def rewrite_query(user_query, chat_history):
    """
    通用版多轮对话改写：将用户的口语化提问改写为适合检索的完整句子
    """
    if not chat_history or len(chat_history) < 1:
        return user_query

    # 只看最近 2 轮对话，避免上下文过长干扰
    recent_history = chat_history[-2:]
    history_text = ""
    for msg in recent_history:
        role = "用户" if msg['role'] == 'user' else "助手"
        history_text += f"{role}: {msg['content']}\n"

    # 【通用化提示词】
    prompt = f"""任务：根据对话历史，将用户的最新提问改写为一个指代清晰、独立完整的搜索语句。
    
    要求：
    1. 补全指代词（如“它”、“这个”、“其”），使其指代上文中的具体对象。
    2. 保持原意，不要回答问题，只需输出改写后的句子。
    
    对话历史：
    {history_text}
    
    用户最新提问：{user_query}
    
    改写结果："""

    try:
        response = dashscope.Generation.call(
            model='qwen-turbo',
            messages=[{'role': 'user', 'content': prompt}],
            result_format='message'
        )
        if response.status_code == 200:
            rewritten = response.output.choices[0].message.content.strip()
            # print(f"🔍 [Query Rewrite] {user_query} -> {rewritten}")
            return rewritten
    except Exception as e:
        print(f"⚠️ 改写失败，使用原问题: {e}")
        
    return user_query

def get_answer_stream(query, db_path, chat_history=[], embedding_model=None):
    """
    核心问答链路：
    1. 改写 -> 2. 混合检索(Vector+BM25) -> 3. 重排序(Rerank) -> 4. 上下文构建 -> 5. 流式生成
    
    Args:
        embedding_model: 必须传入已加载的 HuggingFaceEmbeddings 对象
    """
    # 0. 检查模型参数
    if embedding_model is None:
        raise ValueError("❌ get_answer_stream 调用错误：必须传入 embedding_model 参数！")

    # 1. 查询改写
    search_query = rewrite_query(query, chat_history)
    print(f"🔍 [正在检索] {search_query}")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"找不到数据库路径: {db_path}")

    # --- 2. 混合检索 (Hybrid Search) ---
    
    # 2.1 向量检索 (Vector Search)
    vector_docs = []
    try:
        # 直接使用传入的 embedding_model，无需重新加载，速度极快
        vectordb = Chroma(persist_directory=db_path, embedding_function=embedding_model)
        # 召回 Top-10 给 Rerank 筛选
        vector_docs = vectordb.similarity_search(search_query, k=10)
    except Exception as e:
        print(f"⚠️ 向量检索异常: {e}")
    
    # 2.2 关键词检索 (BM25)
    bm25_docs = []
    bm25_path = os.path.join(db_path, "bm25_data.pkl")
    if os.path.exists(bm25_path):
        try:
            with open(bm25_path, "rb") as f:
                data = pickle.load(f)
                # 实时构建检索器 (内存操作，很快)
                bm25_retriever = BM25Retriever.from_texts(
                    texts=data["documents"], 
                    metadatas=data["metadatas"]
                )
                bm25_retriever.k = 10
                bm25_docs = bm25_retriever.get_relevant_documents(search_query)
        except Exception as e:
            print(f"⚠️ BM25 读取失败: {e}")

    # 2.3 结果融合与去重
    combined_docs = vector_docs + bm25_docs
    unique_docs = []
    seen_content = set()
    
    for doc in combined_docs:
        # 简单去重：内容完全一致则跳过
        if doc.page_content not in seen_content:
            unique_docs.append(doc)
            seen_content.add(doc.page_content)

    # --- 3. 重排序 (Rerank) ---
    # 从融合结果中选出最相关的 Top-3
    final_docs = rerank_documents(search_query, unique_docs, top_k=3)
    
    # --- 4. 构建上下文 (Context) ---
    context_list = []
    for doc in final_docs:
        # 尝试获取页码，如果没有则显示 ?
        page = doc.metadata.get('source_page', '?')
        context_list.append(f"[参考来源 - 第{page}页]\n{doc.page_content}")
    
    context_str = "\n\n".join(context_list)

    # --- 5. 构建通用版系统提示词 (System Prompt) ---
    system_prompt = f"""你是一个专业的文档智能分析助手。请严格根据用户上传的【参考资料】来回答问题。

【回答核心原则】
1. **依据原文**：所有答案必须从上下文中提取，严禁使用你自带的外部知识进行编造。
2. **客观中立**：如果参考资料中没有提到相关内容，请直接回答：“抱歉，当前文档中未找到关于此问题的描述。”
3. **来源标注**：在回答的关键信息后，请尽量用括号标注来源页码，例如 (P5)。

【排版要求】
1. **结构化**：请优先使用 Markdown 列表 (Bullet Points) 来组织答案，使其清晰易读。
2. **重点高亮**：对关键的数据、结论、实体名称，请使用 **加粗** 标记。

【参考资料】
{context_str}
"""

    # --- 6. 调用大模型 (Qwen) ---
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': query} # 这里使用原始 query 保持用户语气，context 已经包含了准确信息
    ]

    responses = dashscope.Generation.call(
        model='qwen-turbo',
        messages=messages,
        result_format='message',
        stream=True,
        incremental_output=True
    )
    
    # 返回：(响应流, 精选出的文档列表)
    return responses, final_docs