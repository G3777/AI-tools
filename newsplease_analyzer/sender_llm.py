import os
import json
from llm_analysis import analyze_data

def process_json_file(file_path):
    """
    处理单个JSON文件，提取cleaned_content并进行分析
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} contains invalid JSON.")
        return None
    
    cleaned_content = data.get("cleaned_content", "")
    if cleaned_content:
        analysis = analyze_data(cleaned_content)
        return analysis
    return None

def process_directory(input_dir, output_dir):
    """
    处理目录中的所有JSON文件
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_dir, file)
                
                analysis = process_json_file(input_file_path)
                if analysis:
                    with open(output_file_path, "w", encoding="utf-8") as f:
                        json.dump({"analysis": analysis}, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    input_directory = "C:\\Users\\ktgr3\\news-please-repo\\data\\processed"
    output_directory = "C:\\Users\\ktgr3\\news-please-repo\\data\\analyzed"
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    process_directory(input_directory, output_directory)
