import newspaper
from concurrent.futures import ThreadPoolExecutor
from newspaper import utils
from datetime import datetime, timedelta
import json
import os

# Enable cache and set cache directory
utils.cache_disk.enabled = True
utils.cache_disk.cache_dir = 'C:\\Users\\ktgr3\\Desktop\\processing\\newspaper_analyzer\\cache'

# Define a function to build newspaper objects
def build_newspaper(url, language='zh'):
    return newspaper.build(url, language=language, memoize_articles=False)

# Define a function to clean content
def clean_content(content):
    return content.strip()

# Define a function to print the URLs and article titles for a given paper
def process_paper_info(paper, name, title_count):
    articles_info = []
    try:
        for article in paper.articles:
            article.url = clean_content(article.url)
            article.download()
            article.parse()
            article.title = clean_content(article.title)
            article.publish_date = article.publish_date or datetime.now()  # Use current date if publish_date is None
            
            # Count the occurrences of each title
            if article.title in title_count:
                title_count[article.title] += 1
            else:
                title_count[article.title] = 1
            
            # Filter articles published within the last month
            one_month_ago = datetime.now() - timedelta(days=30)
            if article.publish_date >= one_month_ago and title_count[article.title] < 2:
                articles_info.append({
                    "title": article.title,
                    "content": article.text,
                    "url": article.url,
                    "publish_date": article.publish_date.strftime('%Y-%m-%d')
                })
    except Exception as e:
        print(f"Error processing {name}: {e}")
    return articles_info

# List of newspaper sources
sources = {
    "sina": "http://finance.sina.com.cn/",
    "caijingCN": "http://www.caijing.com.cn/",
    "eastmoney": "http://finance.eastmoney.com/",
    "huitong": "http://www.fx678.com/"
}

# Create a ThreadPoolExecutor to handle concurrent fetching of articles
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = []
    title_count = {}
    all_articles_info = []
    for name, url in sources.items():
        paper = build_newspaper(url)
        futures.append(executor.submit(process_paper_info, paper, name, title_count))

    # Wait for all futures to complete and collect results
    for future in futures:
        all_articles_info.extend(future.result())

# Save the collected articles info to a JSON file
output_path = 'C:\\Users\\ktgr3\\Desktop\\processing\\newspaper_analyzer\\cache\\articles_info.json'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(all_articles_info, f, ensure_ascii=False, indent=4)

print(f"Articles info saved to {output_path}")
