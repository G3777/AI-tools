import os
import re
import json
from bs4 import BeautifulSoup

def clean_content(content):
    """
    清理内容，去除空白、无效信息
    """
    # 使用BeautifulSoup解析HTML
    soup = BeautifulSoup(content, 'html.parser')
    
    # 去除脚本和样式标签
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()
    
    # 获取纯文本内容
    text = soup.get_text()
    
    # 去除多余的空白字符
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 去除头部和尾部无关内容
    text = remove_unwanted_sections(text)
    
    return text

def remove_unwanted_sections(text):
    """
    去除头部和尾部无关内容
    """
    # 定义头部和尾部的标记
    header_marker = "新浪首页"
    footer_marker = "相关新闻 返回顶部"
    
    # 找到头部和尾部标记的位置
    header_pos = text.find(header_marker)
    footer_pos = text.find(footer_marker)
    
    # 去除头部和尾部内容
    if header_pos != -1:
        text = text[header_pos + len(header_marker):]
    if footer_pos != -1:
        text = text[:footer_pos]
    
    # 去除广告内容
    text = re.sub(r'广告.*?新浪', '', text)
    
    return text

def is_relevant(content):
    """
    判断内容是否相关
    """
    # 这里可以根据具体需求定义相关性的判断标准
    # 例如：内容长度大于50个字符
    return len(content) > 50

def process_html_file(file_path):
    """
    处理单个HTML文件，提取、清理和过滤内容
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="gbk") as f:
            raw_content = f.read()
    
    cleaned_content = clean_content(raw_content)
    if cleaned_content and is_relevant(cleaned_content):
        return {"cleaned_content": cleaned_content}
    return None

def process_directory(input_dir, output_dir):
    """
    处理目录中的所有HTML文件
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".html"):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_dir, os.path.splitext(file)[0] + ".json")
                
                processed_data = process_html_file(input_file_path)
                if processed_data:
                    with open(output_file_path, "w", encoding="utf-8") as f:
                        json.dump(processed_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    input_directory = "C:\\Users\\ktgr3\\news-please-repo\\data"
    output_directory = "C:\\Users\\ktgr3\\news-please-repo\\data\\processed"
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    process_directory(input_directory, output_directory)
