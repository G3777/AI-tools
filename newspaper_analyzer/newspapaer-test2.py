import newspaper
from newspaper.mthreading import fetch_news
from newspaper import utils

utils.cache_disk.enabled = False

sina_paper = newspaper.build('http://finance.sina.com.cn/', language='zh', memoize_articles=False)
caijingCN_paper = newspaper.build('http://www.caijing.com.cn/', language='zh', memoize_articles=False)
eastmoney_paper = newspaper.build('http://finance.eastmoney.com/', language='zh', memoize_articles=False)
huitong_paper = newspaper.build('http://www.fx678.com/', language='zh', memoize_articles=False)

# cnn_paper = newspaper.build('http://www.cnn.com/business', memoize_articles=False, number_threads=2)
# foxnews_paper = newspaper.build('http://www.foxbusiness.com', memoize_articles=False, number_threads=2)

sina_paper.size()
caijingCN_paper.size()
eastmoney_paper.size()
huitong_paper.size()

papers = [sina_paper, caijingCN_paper, eastmoney_paper, huitong_paper]

def print_paper_urls(paper, name):
    print(f"{name} category url:\n", paper.category_urls())
    print(f"{name} feed url:\n", paper.feed_urls())
    article_urls = [article.url for article in paper.articles]
    print(f"{name} article url:\n", article_urls)

    article_titles = [article.title for article in paper.articles]
    print(f"{name} article title:\n", article_titles)

for paper in papers:
    if paper == sina_paper:
        print_paper_urls(sina_paper, "sina")
    elif paper == caijingCN_paper:
        print_paper_urls(caijingCN_paper, "caijingCN")
    elif paper == eastmoney_paper:
        print_paper_urls(eastmoney_paper, "eastmoney")
    elif paper == huitong_paper:
        print_paper_urls(huitong_paper, "huitong")




# fetched_news = fetch_news(papers, threads=4)

#-------------------------------------------------
# news_urls = [
#     "http://finance.sina.com.cn/",
#     "http://www.caijing.com.cn/",
#     "http://finance.eastmoney.com/",
#     "http://www.fx678.com/",
# ]
# fetched_news = fetch_news(news_urls, threads=4)
# # print(fetched_news)
#-------------------------------------------------


