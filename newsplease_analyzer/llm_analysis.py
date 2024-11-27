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


# import openai
# import requests

# # Configure OpenAI API settings
# openai.api_key = "not-needed"  # API key not needed for local models
# primary_url = "http://192.168.1.6:1234/v1"
# fallback_url = "http://localhost:1234/v1"

# def set_api_base(url):
#     """Set the OpenAI API base URL."""
#     openai.api_base = url

# def check_server(url):
#     """Check if the server is reachable."""
#     try:
#         response = requests.get(url)
#         return response.status_code == 200
#     except requests.ConnectionError:
#         return False

# def analyze_data(data):
#     """Analyze data using the OpenAI ChatCompletion API."""
#     response = openai.ChatCompletion.create(
#         model="local-model",
#         messages=[{"role": "assistant", "content": data}]
#     )
#     analysis = response.choices[0].message['content']
#     # analysis = response.choices[0]['message']['content']
#     return analysis

# # Set the API base based on server availability
# if check_server(primary_url):
#     set_api_base(primary_url)
# else:
#     print(f"Primary server {primary_url} is unavailable, switching to fallback.")
#     set_api_base(fallback_url)
