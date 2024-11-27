# import os
# import json
# import openai

# # 配置LM Studio的API
# openai.api_base = "http://localhost:1234/v1"
# openai.api_key = "not-needed"

# def analyze_data(data):
#     response = openai.ChatCompletion.create(
#         model="local-model",
#         messages=[{"role": "assistant", "content": data}]
#     )
#     analysis = response.choices[0].message['content']
#     return analysis

# def send_news_to_llm(directory):
#     for filename in os.listdir(directory):
#         if filename.endswith(".json"):
#             filepath = os.path.join(directory, filename)
#             with open(filepath, 'r', encoding='utf-8') as file:
#                 news_list = json.load(file)
#                 for news_data in news_list:
#                     title = news_data.get('title', 'No Title')
#                     content = news_data.get('content', 'No Content')
#                     data = f"Title: {title}\nContent: {content}"
#                     analysis = analyze_data(data)
#                     print(f"Analysis for {filename}:\n{analysis}\n")

# if __name__ == "__main__":
    # input_directory = r"C:\\Users\\ktgr3\Desktop\\processing\\newspaper_analyzer\\cache"
    # output_file = r"C:\\Users\\ktgr3\Desktop\\processing\\newspaper_analyzer\\analyzed\\article_analyzed.json"
    # send_news_to_llm(input_directory)


import os
import json
import openai

# 配置LM Studio的API
openai.api_base = "http://localhost:1234/v1"
openai.api_key = "not-needed"

def analyze_data(data):
    response = openai.ChatCompletion.create(
        model="local-model",
        messages=[{"role": "assistant", "content": data}]
    )
    analysis = response.choices[0].message['content']
    return analysis

def process_json_file(file_path, output_file):
    """
    处理单个JSON文件，提取title和content并进行分析，并将结果写入输出文件
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} contains invalid JSON.")
        return
    
    with open(output_file, "a", encoding="utf-8") as f:
        for data in data_list:
            if isinstance(data, dict):  # 确保data是一个字典
                title = data.get("title", "No Title")
                content = data.get("content", "No Content")
                data_to_analyze = f"Title: {title}\nContent: {content}"
                analysis = analyze_data(data_to_analyze)
                analyzed_article = {
                    "title": title,
                    "content": content,
                    "analysis": analysis
                }
                json.dump(analyzed_article, f, ensure_ascii=False, indent=4)
                f.write("\n")

def process_directory(input_dir, output_file):
    """
    处理目录中的所有JSON文件
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                input_file_path = os.path.join(root, file)
                process_json_file(input_file_path, output_file)

if __name__ == "__main__":
    input_directory = r"C:\\Users\\ktgr3\Desktop\\processing\\newspaper_analyzer\\cache"
    output_file = r"C:\\Users\\ktgr3\Desktop\\processing\\newspaper_analyzer\\analyzed\\news_analyzed.json"
    
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    
    # 清空输出文件内容
    open(output_file, 'w').close()
    
    process_directory(input_directory, output_file)

