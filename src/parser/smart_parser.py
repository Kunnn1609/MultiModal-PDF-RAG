import re
import fitz  # PyMuPDF
import numpy as np
import cv2
import logging
from paddleocr import PaddleOCR

# 屏蔽 PaddleOCR 的调试日志，保持控制台整洁
logging.getLogger("ppocr").setLevel(logging.WARNING)

def is_text_garbled_or_empty(text, min_length=15):
    """
    启发式规则：更智能地判断提取的文本是否为乱码或内容过少
    """
    clean_text = text.replace(" ", "").replace("\n", "").strip()
    
    # 1. 如果完全没有内容，必须 OCR
    if not clean_text:
        return True

    # 2. 检查 CID 乱码 (这是 PDF 字体缺失最显著的特征)
    cid_pattern = re.compile(r'\(cid:\d+\)')
    if len(cid_pattern.findall(text)) > 5:
        print(f"   [检测] 发现大量 (cid) 编码，判定为乱码。")
        return True

    # 3. 检查中文占比
    # 学术论文通常含有大量汉字。如果提取结果有汉字，说明提取链路基本正常
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in clean_text)
    
    # 如果字数很少且没有中文（排除掉页码或Logo等小块提取物）
    if len(clean_text) < min_length and not has_chinese:
        return True
    
    # 4. 检查非法字符比例 (如 ▯, ?, )
    # 如果异常字符超过 30%，判定为乱码
    bad_chars = len(re.findall(r'[▯\?]', clean_text))
    if len(clean_text) > 0 and (bad_chars / len(clean_text)) > 0.3:
        return True

    return False

def ocr_page_image(page, ocr_engine):
    """
    将页面转为图片并进行 OCR
    """
    print("   [OCR] 启动视觉识别中...")
    
    # 渲染高分辨率图片
    zoom = 2
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    
    # 转换为 OpenCV 格式
    img_data = np.frombuffer(pix.tobytes("png"), dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

    # 调用新版 predict API
    result = ocr_engine.predict(img)
    
    ocr_text = ""
    if result:
        # 兼容列表返回格式：[[坐标, (文本, 置信度)], ...]
        if isinstance(result, list):
            for line in result:
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    # 获取文本部分
                    content = line[1]
                    if isinstance(content, (list, tuple)) and len(content) > 0:
                        ocr_text += str(content[0]) + "\n"
                    elif isinstance(content, str):
                        ocr_text += content + "\n"
        # 兼容字典返回格式
        elif isinstance(result, dict) and 'rec_text' in result:
            ocr_text = result['rec_text']
            
    return ocr_text

def smart_extract(pdf_path, ocr_engine):
    """
    主解析逻辑：自适应切换直接提取与 OCR
    """
    doc = fitz.open(pdf_path)
    full_content = []

    print(f"🚀 开始智能解析: {pdf_path}")

    for page_num, page in enumerate(doc):
        # 1. 尝试直接获取文本
        raw_text = page.get_text().strip()
        
        # 2. 判断是否满足 OCR 触发条件
        # 条件：1. 提取不到任何文本；2. 触发乱码或过短规则
        need_ocr = False
        if not raw_text:
            need_ocr = True
            reason = "无文本流（可能是纯图片）"
        elif is_text_garbled_or_empty(raw_text):
            need_ocr = True
            reason = "检测到乱码或无效短文本"
        
        # 3. 执行提取
        if need_ocr:
            print(f"📄 第 {page_num + 1} 页: ⚠️ {reason}，执行 OCR...")
            final_text = ocr_page_image(page, ocr_engine)
            method = "OCR"
        else:
            print(f"📄 第 {page_num + 1} 页: ✅ 文本提取成功")
            final_text = raw_text
            method = "Direct"

        full_content.append({
            "page_number": page_num + 1,
            "content": final_text,
            "method": method
        })

    doc.close()
    return full_content

if __name__ == "__main__":
    # 初始化引擎（全局只加载一次，节省显存/内存）
    print("⏳ 初始化 PaddleOCR 引擎...")
    engine = PaddleOCR(lang="ch", use_angle_cls=True)

    # 测试文件路径
    test_pdf = r"D:\workspace\finale_workspace\PDF_RAG_Project\data\raw\基于混合RAG的铁路多模态知识库问答系统构建方法_洪英杰.pdf"
    
    results = smart_extract(test_pdf, engine)

    print("\n" + "="*30)
    print("💡 解析结果摘要：")
    for res in results:
        method_tag = "[OCR]" if res['method'] == "OCR" else "[Direct]"
        preview = res['content'][:50].replace('\n', ' ')
        print(f"第 {res['page_number']} 页 {method_tag}: {preview}...")