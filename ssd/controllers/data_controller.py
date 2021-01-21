import pickle
from icrawler.builtin import BingImageCrawler
from .loader import PascalVocLoader


class DataController:

    @classmethod
    def crawl_images(cls):
        crawler = BingImageCrawler(
            downloader_threads=4,
            storage={'root_dir': 'data'})
        crawler.crawl(
            keyword="請求書",
            max_num=1000)

    @classmethod
    def make_train_pkl(cls):
        # pascalvocのxmlファイルをロードしてpklにdump
        loader = PascalVocLoader(
          dir_path='./data/train',
          labels=['company', 'amount']
        )
        loader.load(dump='./data/train/train.pkl')

    @classmethod
    def load_train_pkl(cls):
        # dumpしたpklを読み込んで表示
        with open('./data/train/train.pkl', 'rb') as f:
            data = pickle.load(f)

        print(data)

    @classmethod
    def make_test_pkl(cls):
        # pascalvocのxmlファイルをロードしてpklにdump
        loader = PascalVocLoader(
          dir_path='./data/test',
          labels=['company', 'amount']
        )
        loader.load(dump='./data/train/train.pkl')
