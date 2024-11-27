import newspaper
from concurrent.futures import ThreadPoolExecutor
from newspaper import utils

# Disable cache to avoid overwhelming the server
utils.cache_disk.enabled = False

# Set up newspaper objects for each source
sina_paper = newspaper.build('http://finance.sina.com.cn/', language='zh', memoize_articles=False)
caijingCN_paper = newspaper.build('http://www.caijing.com.cn/', language='zh', memoize_articles=False)
eastmoney_paper = newspaper.build('http://finance.eastmoney.com/', language='zh', memoize_articles=False)
huitong_paper = newspaper.build('http://www.fx678.com/', language='zh', memoize_articles=False)

# Function to clean URLs
def clean_content(content):
    return content.strip()

# Define a function to print the URLs and article titles for a given paper
def print_paper_info(paper, name):
    try:
        # Print category, feed, and article URLs
        # print(f"{name} category url:\n", paper.category_urls())
        print(f"{name} feed url:\n", paper.feed_urls())
        article_urls = [clean_content(article.url) for article in paper.articles]
        print(f"{name} article url:\n", article_urls)

        # Print article titles and content
        for article in paper.articles:
            article.url = clean_content(article.url)
            article.download()
            article.parse()
            article.title = clean_content(article.title)
            print(f"Title: {article.title}")
            print(f"Content: {article.text}\n")
    except Exception as e:
        print(f"Error processing {name}: {e}")

# Create a ThreadPoolExecutor to handle concurrent fetching of articles
with ThreadPoolExecutor(max_workers=10) as executor:
    # Fetch and print articles for each paper using the executor
    papers = [sina_paper, caijingCN_paper, eastmoney_paper, huitong_paper]
    futures = []
    for paper in papers:
        if paper == sina_paper:
            futures.append(executor.submit(print_paper_info, paper, "sina"))
        elif paper == caijingCN_paper:
            futures.append(executor.submit(print_paper_info, paper, "caijingCN"))
        elif paper == eastmoney_paper:
            futures.append(executor.submit(print_paper_info, paper, "eastmoney"))
        elif paper == huitong_paper:
            futures.append(executor.submit(print_paper_info, paper, "huitong"))

    # Wait for all futures to complete
    for future in futures:
        future.result()
