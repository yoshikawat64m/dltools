import pickle

from ssd.pascal_voc_loader import PascalVocLoader


def make_train_pkl():
    # pascalvocのxmlファイルをロードしてpklにdump
    loader = PascalVocLoader(
      dir_path='./data/train',
      labels=['company', 'amount']
    )
    loader.load(dump='./data/train/train.pkl')


def load_train_pkl():
    # dumpしたpklを読み込んで表示
    with open('./data/train/train.pkl', 'rb') as f:
        data = pickle.load(f)

    print(data)


def make_test_pkl():
    # pascalvocのxmlファイルをロードしてpklにdump
    loader = PascalVocLoader(
      dir_path='./data/test',
      labels=['company', 'amount']
    )
    loader.load(dump='./data/train/train.pkl')
