import dashscope

def evaluate_response(question, context, answer):
    """
    使用 LLM 对 RAG 的回答进行量化评分
    """
    eval_prompt = f"""你是一名严格的 AI 问答评审专家。请根据以下提供的上下文和回答，进行多维度打分（0-10分）。

### 原始问题 ###
{question}

### 检索到的上下文 ###
{context}

### AI 生成的回答 ###
{answer}

### 评分要求 ###
1. Faithfulness (忠实度)：回答是否仅根据上下文得出，无误导性信息。
2. Answer Relevance (相关性)：回答是否直接且准确地回答了问题。
3. Context Support (支撑度)：上下文是否足以支撑该回答。

请按以下 JSON 格式输出结果：
{{
  "faithfulness": 分数,
  "relevance": 分数,
  "support": 分数,
  "reason": "简短的评分理由"
}}
"""

    response = dashscope.Generation.call(
        model='qwen-max',  # 建议评分用更高阶的模型
        messages=[{'role': 'system', 'content': '你是一个客观的评审员'},
                  {'role': 'user', 'content': eval_prompt}],
        result_format='message'
    )
    
    return response.output.choices[0].message.content