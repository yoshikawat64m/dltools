from icrawler.builtin import BingImageCrawler


bing_crawler = BingImageCrawler(
    downloader_threads=4,
    storage={'root_dir': 'data2'})


bing_crawler.crawl(
    keyword="請求書",
    max_num=1000)
