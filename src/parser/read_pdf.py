import fitz  # PyMuPDF
import os


def simple_pdf_read(pdf_path):
    # 打开 PDF 文件
    doc = fitz.open(pdf_path)
    print(f"📄 正在读取: {pdf_path}")
    print(f"📊 总页数: {doc.page_count}")

    # 尝试读取第一页
    page = doc[0]
    text = page.get_text()

    print("\n--- 第一页内容预览 ---")
    if text.strip():
        print(text[:200] + "...")  # 只打印前200个字符
    else:
        print("⚠️ 警告: 第一页无法提取文本，可能需要 OCR！")

    doc.close()


if __name__ == "__main__":
    # 确保您在 data/raw/ 下放了一个测试用的 pdf 文件
    # 将 'test.pdf' 替换为您的实际文件名
    pdf_path = r"D:\workspace\finale_workspace\PDF_RAG_Project\data\raw\基于混合RAG的铁路多模态知识库问答系统构建方法_洪英杰.pdf"

    # 简单的路径检查
    if os.path.exists(pdf_path):
        simple_pdf_read(pdf_path)
    else:
        print("❌ 错误: 找不到文件，请在 data/raw/ 文件夹下放入一个 PDF 文件。")