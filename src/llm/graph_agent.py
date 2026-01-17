import json
import dashscope
from streamlit_agraph import agraph, Node, Edge, Config

def extract_triplets_from_text(text):
    """
    调用大模型提取文本中的实体关系三元组
    """
    if not text:
        return []
        
    prompt = f"""
    你是一个知识图谱专家。请从下面的文本中提取核心的“实体-关系-实体”三元组。
    
    【任务要求】
    1. 提取 3 到 5 组核心关系。
    2. 实体要简短（如“CRH380A”、“受电弓”）。
    3. 必须输出严格的 JSON 格式列表，不要包含 Markdown 标记（如 ```json）。
    4. 格式示例：[["实体A", "关系", "实体B"], ["实体B", "属性", "值"]]

    【待分析文本】
    {text[:800]} 
    
    【JSON结果】
    """
    
    try:
        response = dashscope.Generation.call(
            model='qwen-turbo',
            messages=[{'role': 'user', 'content': prompt}],
            result_format='message'
        )
        
        if response.status_code == 200:
            content = response.output.choices[0].message.content
            # 清理可能的 markdown 符号
            content = content.replace("```json", "").replace("```", "").strip()
            triplets = json.loads(content)
            return triplets
    except Exception as e:
        print(f"Graph extraction failed: {e}")
        return []
    return []

def build_graph_config(triplets):
    """
    将三元组转换为 streamlit-agraph 需要的节点和边
    """
    nodes = set()
    edges = []
    
    added_nodes = set()
    
    for src, rel, dst in triplets:
        # 添加源节点
        if src not in added_nodes:
            nodes.add(Node(id=src, label=src, size=20, color="#FF6B6B")) # 红色
            added_nodes.add(src)
        
        # 添加目标节点
        if dst not in added_nodes:
            nodes.add(Node(id=dst, label=dst, size=15, color="#4ECDC4")) # 青色
            added_nodes.add(dst)
            
        # 添加边
        edges.append(Edge(source=src, target=dst, label=rel))
        
    config = Config(width="100%", height=400, directed=True, 
                    nodeHighlightBehavior=True, highlightColor="#F7A7A6",
                    collapsible=False)
                    
    return list(nodes), edges, config