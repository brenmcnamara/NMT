from utils import load_en_fr

if __name__ == '__main__':
    (train, valid), EN, FR = load_en_fr()
    print(len(train), len(valid))
