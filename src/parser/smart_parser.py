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
    # 例如：(cid:1234) 这种格式
    if len(re.findall(r'\(cid:\d+\)', text)) > 5:
        # print(f"   [检测] 发现大量 (cid) 编码，判定为乱码。")
        return True

    # 3. 检查中文占比
    # 学术论文通常含有大量汉字。如果提取结果有汉字，说明提取链路基本正常
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in clean_text)
    
    # 如果字数很少且没有中文（排除掉页码或Logo等小块提取物）
    if len(clean_text) < min_length and not has_chinese:
        return True
    
    # 4. 检查非法字符比例 (如 ▯, ?)
    bad_chars = len(re.findall(r'[▯\?]', clean_text))
    if len(clean_text) > 0 and (bad_chars / len(clean_text)) > 0.3:
        return True

    return False

def clean_header_footer(text):
    """
    【新增功能】清洗页眉、页脚和出版元数据噪音
    保留摘要（Abstract），但过滤掉干扰阅读的版面信息
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        content = line.strip()
        if not content:
            continue
            
        # 1. 过滤纯数字 (通常是页码，如 "1", "45")
        if content.isdigit() and len(content) < 5:
            continue
            
        # 2. 过滤常见的页眉/出版信息特征
        # 规则：长度较短(小于80字符) 且 包含特定关键词
        # 这些词通常出现在页眉页脚，而不是正文中
        is_header_footer = False
        if len(content) < 100:
            noise_keywords = [
                "ISSN", "DOI", "http", "www.", "cnki", 
                "学报", "Journal", "Vol.", "No.", "期", "卷",
                "网络首发", "引用格式", "Computer Science", "Page"
            ]
            if any(k in content for k in noise_keywords):
                is_header_footer = True
        
        if not is_header_footer:
            cleaned_lines.append(line)
            
    return "\n".join(cleaned_lines)

def ocr_page_image(page, ocr_engine):
    """
    将页面转为图片并进行 OCR
    """
    print("   [OCR] 启动视觉识别中...")
    
    # 渲染高分辨率图片 (zoom=2 保证清晰度)
    zoom = 2
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    
    # 转换为 OpenCV 格式供 PaddleOCR 使用
    img_data = np.frombuffer(pix.tobytes("png"), dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

    # 调用 OCR
    result = ocr_engine.predict(img)
    
    ocr_text = ""
    if result:
        # 兼容 PaddleOCR 不同的返回格式 (list or dict)
        if isinstance(result, list):
            for line in result:
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    content = line[1]
                    if isinstance(content, (list, tuple)) and len(content) > 0:
                        ocr_text += str(content[0]) + "\n"
                    elif isinstance(content, str):
                        ocr_text += content + "\n"
        elif isinstance(result, dict) and 'rec_text' in result:
            ocr_text = result['rec_text']
            
    return ocr_text

def smart_extract(pdf_path, ocr_engine):
    """
    主解析逻辑：
    1. 尝试直接提取 -> 失败则 OCR
    2. 页眉页脚清洗 (保留摘要)
    3. 参考文献截断 (防止语义污染)
    """
    doc = fitz.open(pdf_path)
    full_content = []
    total_pages = len(doc)
    
    # 🛑 参考文献截断标志位
    stop_parsing = False 

    print(f"🚀 开始智能解析: {pdf_path} (共 {total_pages} 页)")

    for page_num, page in enumerate(doc):
        # 0. 如果已经触发了截断机制，直接跳过剩余页面
        if stop_parsing:
            print(f"🛑 [截断] 跳过第 {page_num + 1} 页 (参考文献/附录区域)。")
            break

        # 1. 尝试直接获取文本
        raw_text = page.get_text().strip()
        
        # 2. 判断是否满足 OCR 触发条件
        need_ocr = False
        method = "Direct"
        
        if not raw_text:
            need_ocr = True
            reason = "无文本流（可能是纯图片）"
        elif is_text_garbled_or_empty(raw_text):
            need_ocr = True
            reason = "检测到乱码或无效短文本"
        
        # 3. 执行提取
        final_text = ""
        if need_ocr:
            print(f"📄 第 {page_num + 1} 页: ⚠️ {reason}，执行 OCR...")
            final_text = ocr_page_image(page, ocr_engine)
            method = "OCR"
        else:
            # print(f"📄 第 {page_num + 1} 页: ✅ 文本提取成功")
            final_text = raw_text

        # 4. 【新增】清洗页眉页脚
        # 在处理参考文献之前先清洗，防止页眉里的关键词干扰判断
        final_text = clean_header_footer(final_text)

        # 5. 【新增】检测参考文献并截断
        # 逻辑：只在文档后半部分检查，防止目录中出现“参考文献”导致误杀
        if page_num > total_pages * 0.5:
            lines = final_text.split('\n')
            cleaned_lines_for_this_page = []
            
            for line in lines:
                # 去除空格后检查关键词
                clean_line = line.strip().replace(" ", "")
                # 检查常见的中英文参考文献标题 (独占一行或极短)
                if clean_line in ["参考文献", "References", "Bibliography", "主要参考文献", "Reference"]:
                    stop_parsing = True
                    print(f"✂️ [检测] 在第 {page_num + 1} 页发现参考文献列表，启动截断。")
                    break 
                cleaned_lines_for_this_page.append(line)
            
            # 如果本页触发了截断，只保留截断前的内容
            if stop_parsing:
                final_text = "\n".join(cleaned_lines_for_this_page)
                # 如果这一页截断后没剩什么内容了，就直接跳过不存
                if not final_text.strip():
                    continue

        # 6. 存入结果
        if final_text.strip():
            full_content.append({
                "page_number": page_num + 1,
                "content": final_text,
                "method": method
            })

    doc.close()
    return full_content

if __name__ == "__main__":
    # 测试代码
    print("⏳ 初始化 PaddleOCR 引擎...")
    engine = PaddleOCR(lang="ch", use_angle_cls=True)

    # 请替换为你本地的测试文件路径
    test_pdf = r"D:\workspace\finale_workspace\PDF_RAG_Project\data\raw\test.pdf"
    
    # 模拟运行
    # results = smart_extract(test_pdf, engine)
    # for res in results:
    #     print(f"--- Page {res['page_number']} ---")
    #     print(res['content'][:200] + "...")