import streamlit as st
from FlagEmbedding import FlagReranker

# 使用 st.cache_resource 确保模型只加载一次，极大提升速度
@st.cache_resource
def get_reranker():
    """
    加载 BAAI/bge-reranker-base 模型
    第一次运行时会自动下载约 1GB 的模型文件
    """
    # use_fp16=True 在显卡上能加速，CPU上会自动回退
    return FlagReranker('BAAI/bge-reranker-base', use_fp16=True)

def rerank_documents(query, docs, top_k=3):
    """
    对检索到的文档进行重排序
    :param query: 用户问题
    :param docs: 向量检索回来的初步文档列表 (LangChain Document 对象)
    :param top_k: 最终保留前几个最相关的
    """
    if not docs:
        return []

    reranker = get_reranker()
    
    # 构造配对数据 [['问题', '文档内容'], ...]
    pairs = [[query, d.page_content] for d in docs]
    
    # 计算相关性得分
    scores = reranker.compute_score(pairs)
    
    # 将文档和分数打包
    # 如果只有1个文档，scores可能是一个float而不是list，做个兼容
    if isinstance(scores, float):
        scores = [scores]
        
    doc_score_pairs = list(zip(docs, scores))
    
    # 按分数从高到低排序
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # 调试打印（可选）
    # for d, s in doc_score_pairs:
    #     print(f"Score: {s:.4f} | Content: {d.page_content[:20]}...")

    # 返回前 k 个文档
    return [doc for doc, score in doc_score_pairs[:top_k]]