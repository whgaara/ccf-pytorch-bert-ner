import os


def anlysis_length(dir_path='data/train_new'):
    max_len = 0
    for f_name in os.listdir(dir_path):
        with open(os.path.join(dir_path, f_name), 'r', encoding='utf-8') as f:
            text = f.readlines()
            text = [x.strip() for x in text if x]
            if len(text) > max_len:
                max_len = len(text)
            if len(text) > 256:
                print(f_name)
    print(max_len)


if __name__ == '__main__':
    anlysis_length('data/eval_new')
