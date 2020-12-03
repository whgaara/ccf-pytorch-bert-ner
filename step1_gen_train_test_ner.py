import os
import pickle
import random

from tqdm import tqdm
from pretrain_config import *
from roberta.common.tokenizers import Tokenizer


def parse_new_data():
    """
    :return:
    [123, 233, 334, 221, 299, ..., ...]
    [ptzf, b-ypcf, i-ypcf, i-ypcf, e-ypcf, e-yplb, ..., pytzf, ...]
    """
    with open(Class2NumFile, 'rb') as f:
        class2num = pickle.load(f)
    # class2num = {'pad': 0, 'ptzf': 1}
    new_train_data = {}
    new_eval_data = {}
    tokenizer = Tokenizer(VocabPath)
    input_path = 'data/train_new'
    eval_path = 'data/eval_new'
    f_train = open(NerCorpusPath, 'a+', encoding='utf-8')
    f_eval = open(NerEvalPath, 'w', encoding='utf-8')
    category_list = []

    for data_file in os.listdir(input_path):
        if '.txt' not in data_file:
            continue
        file_num = data_file.split('.')[0]
        f1 = open(os.path.join(input_path, data_file), 'r', encoding='utf-8')
        lines = f1.readlines()
        lines = [x.strip().replace(',', '，') for x in lines if x][:-1]
        new_train_data[file_num] = {}
        new_train_data[file_num]['sentence'] = ''
        new_train_data[file_num]['tokens_id'] = []
        new_train_data[file_num]['tokens_class'] = []
        new_train_data[file_num]['tokens_class_num'] = []
        for i, line in enumerate(lines):
            try:
                ch, label = tuple(line.lower().split(' '))
            except:
                print(file_num)
                print(i)
                print(line)
                print('\n')
                ch = '，'
                label = 'o'
            new_train_data[file_num]['sentence'] += ch
            new_train_data[file_num]['tokens_id'].append(tokenizer.token_to_id(ch))
            if label == 'o' or label == '0':
                token_class = 'ptzf'
                token_class_num = 1
            else:
                token_class = label.lower().replace('-', '')
                if token_class[1:] in ['qq', 'vx', 'mobile', 'email']:
                    token_class = 'ptzf'
                if token_class != 'ptzf':
                    category_list.append(token_class[1:])
                if token_class in class2num:
                    token_class_num = class2num[token_class]
                else:
                    token_class_num = len(class2num)
                    class2num[token_class] = token_class_num
            new_train_data[file_num]['tokens_class'].append(token_class)
            new_train_data[file_num]['tokens_class_num'].append(token_class_num)

    for data_file in os.listdir(eval_path):
        if '.txt' not in data_file:
            continue
        file_num = data_file.split('.')[0]
        f1 = open(os.path.join(eval_path, data_file), 'r', encoding='utf-8')
        lines = f1.readlines()
        lines = [x.strip().replace(',', '，') for x in lines if x][:-1]
        new_eval_data[file_num] = {}
        new_eval_data[file_num]['sentence'] = ''
        new_eval_data[file_num]['tokens_id'] = []
        new_eval_data[file_num]['tokens_class'] = []
        new_eval_data[file_num]['tokens_class_num'] = []
        for i, line in enumerate(lines):
            try:
                ch, label = tuple(line.lower().split(' '))
            except:
                print(file_num)
                print(i)
                print(line)
                print('\n')
                ch = '，'
                label = 'o'
            new_eval_data[file_num]['sentence'] += ch
            new_eval_data[file_num]['tokens_id'].append(tokenizer.token_to_id(ch))
            if label == 'o':
                token_class = 'ptzf'
                token_class_num = 1
            else:
                token_class = label.lower().replace('-', '')
                if token_class[1:] in ['qq', 'vx', 'mobile', 'email']:
                    token_class = 'ptzf'
                if token_class != 'ptzf':
                    category_list.append(token_class[1:])
                token_class_num = class2num[token_class]
            new_eval_data[file_num]['tokens_class'].append(token_class)
            new_eval_data[file_num]['tokens_class_num'].append(token_class_num)

    print(set(category_list))

    # 补全所有的句子
    for num in new_train_data:
        difference = SentenceLength - len(new_train_data[num]['sentence'])
        new_train_data[num]['tokens_id'].extend([0] * difference)
        new_train_data[num]['tokens_class'].extend(['pad'] * difference)
        new_train_data[num]['tokens_class_num'].extend([class2num['pad']] * difference)
        new_train_data[num]['tokens_id'] = [str(x) for x in new_train_data[num]['tokens_id']]
        new_train_data[num]['tokens_class_num'] = [str(x) for x in new_train_data[num]['tokens_class_num']]
    for num in new_eval_data:
        difference = SentenceLength - len(new_eval_data[num]['sentence'])
        new_eval_data[num]['tokens_id'].extend([0] * difference)
        new_eval_data[num]['tokens_class'].extend(['pad'] * difference)
        new_eval_data[num]['tokens_class_num'].extend([class2num['pad']] * difference)
        new_eval_data[num]['tokens_id'] = [str(x) for x in new_eval_data[num]['tokens_id']]
        new_eval_data[num]['tokens_class_num'] = [str(x) for x in new_eval_data[num]['tokens_class_num']]

    # 将类型及编号进行存储
    # with open(Class2NumFile, 'wb') as f:
    #     pickle.dump(class2num, f)

    for num in new_train_data:
        if new_train_data[num]['sentence']:
            if new_train_data[num]['sentence']:
                f_train.write(new_train_data[num]['sentence'] + ',' +
                              ' '.join(new_train_data[num]['tokens_id']) + ',' +
                              ' '.join(new_train_data[num]['tokens_class']) + ',' +
                              ' '.join(new_train_data[num]['tokens_class_num']) + '\n'
                              )
    for num in new_eval_data:
        if new_eval_data[num]['sentence']:
            if new_eval_data[num]['sentence']:
                f_eval.write(new_eval_data[num]['sentence'] + ',' +
                             ' '.join(new_eval_data[num]['tokens_id']) + ',' +
                             ' '.join(new_eval_data[num]['tokens_class']) + ',' +
                             ' '.join(new_eval_data[num]['tokens_class_num']) + '\n'
                             )


def parse_source_data():
    """
    :return:
    [123, 233, 334, 221, 299, ..., ...]
    [ptzf, b-ypcf, i-ypcf, i-ypcf, e-ypcf, e-yplb, ..., pytzf, ...]
    """
    MaxLen = 0
    class2num = {'pad': 0, 'ptzf': 1}
    total_data = {}
    tokenizer = Tokenizer(VocabPath)
    input_path = os.path.join(NerSourcePath, 'data')
    label_path = os.path.join(NerSourcePath, 'label')
    f_train = open(NerCorpusPath, 'w', encoding='utf-8')
    # f_eval = open(NerEvalPath, 'w', encoding='utf-8')
    category_list = []

    relabel_list = []
    for data_file in os.listdir(input_path):
        label_word_pool = {}
        if '.txt' not in data_file:
            continue
        file_num = data_file.split('.')[0]
        f1 = open(os.path.join(input_path, data_file), 'r', encoding='utf-8')
        f2 = open(os.path.join(label_path, file_num+'.csv'), 'r', encoding='utf-8')
        sentence = f1.read().strip().replace(',', '，')

        # 初始化数据结构
        total_data[int(file_num)] = {}
        total_data[int(file_num)]['sentence'] = sentence
        total_data[int(file_num)]['tokens_id'] = [0] * len(sentence)
        total_data[int(file_num)]['tokens_class'] = ['ptzf'] * len(sentence)
        total_data[int(file_num)]['tokens_class_num'] = [1] * len(sentence)

        # 存储原句tokenid, 101表示cls
        for i, token in enumerate(sentence):
            id = tokenizer.token_to_id(token)
            if not id:
                print('警告！本地vocab缺少以下字符：%s！' % token)
                print(sentence)
                # 100表示UNK
                total_data[int(file_num)]['tokens_id'][i] = 100
            else:
                total_data[int(file_num)]['tokens_id'][i] = id
        label_lines = f2.readlines()[1:]
        for label_line in label_lines:
            label_line = label_line.split(',', 4)
            assert len(label_line) == 5
            category = label_line[1]
            begin = int(label_line[2])
            end = int(label_line[3])
            label_words = label_line[4].strip()
            category_list.append(category)

            # if '启示录》' in label_words:
            #     x = 1
            # if category == 'organization':
            #     print(file_num, label_words)

            # 校验标记正确性
            ori_words = sentence[begin:end+1]
            if ori_words != label_words:
                print('标记位置错误：%s，%s！' % (file_num, label_words))

            # 校验重复标记
            for j in range(begin, end+1):
                if j in label_word_pool:
                    relabel_list.append(file_num)
                else:
                    label_word_pool[j] = 'ok'

            if category in ['QQ', 'vx', 'mobile', 'email']:
                continue
            if begin == end:
                if 'b' + category not in class2num:
                    class2num['b' + category] = len(class2num)
                total_data[int(file_num)]['tokens_class'][end] = 'b' + category
                total_data[int(file_num)]['tokens_class_num'][end] = class2num['b' + category]
            if end - begin > 0:
                if 'b' + category not in class2num:
                    class2num['b' + category] = len(class2num)
                if 'i' + category not in class2num:
                    class2num['i' + category] = len(class2num)
                total_data[int(file_num)]['tokens_class'][begin] = 'b' + category
                total_data[int(file_num)]['tokens_class'][begin+1:end] = ['i' + category] * (end - begin)
                total_data[int(file_num)]['tokens_class_num'][begin] = class2num['b' + category]
                total_data[int(file_num)]['tokens_class_num'][begin+1:end] = [class2num['i' + category]] * (end - begin)

    # 将长句进行分割
    new_total_data = {}
    tmp_docker = ['', [], [], []]
    for num in total_data:
        if len(total_data[num]['sentence']) <= SentenceLength:
            tl = len(new_total_data)
            new_total_data[tl] = {}
            new_total_data[tl]['sentence'] = total_data[num]['sentence']
            new_total_data[tl]['tokens_id'] = total_data[num]['tokens_id']
            new_total_data[tl]['tokens_class'] = total_data[num]['tokens_class']
            new_total_data[tl]['tokens_class_num'] = total_data[num]['tokens_class_num']
            tmp_docker = ['', [], [], []]
        else:
            ts = list(total_data[num]['sentence'])
            ti = total_data[num]['tokens_id']
            tc = total_data[num]['tokens_class']
            tn = total_data[num]['tokens_class_num']
            for i, word in enumerate(ts):
                if word in [',', '，', '。', '?', '？', '!', '！', '~', ':', '：']:
                    if len(tmp_docker[0]) > MaxLen:
                        MaxLen = len(tmp_docker[0])
                    if len(tmp_docker[0]) > 200:
                        x = 1
                    if tc[i][0] == 'i' or 0 < len(tmp_docker[0]) < 10:
                        tmp_docker[0] += word
                        tmp_docker[1].append(ti[i])
                        tmp_docker[2].append(tc[i])
                        tmp_docker[3].append(tn[i])
                    else:
                        tl = len(new_total_data)
                        new_total_data[tl] = {}
                        new_total_data[tl]['sentence'] = tmp_docker[0]
                        new_total_data[tl]['tokens_id'] = tmp_docker[1]
                        new_total_data[tl]['tokens_class'] = tmp_docker[2]
                        new_total_data[tl]['tokens_class_num'] = tmp_docker[3]
                        tmp_docker = ['', [], [], []]
                        continue
                else:
                    tmp_docker[0] += word
                    tmp_docker[1].append(ti[i])
                    tmp_docker[2].append(tc[i])
                    tmp_docker[3].append(tn[i])

    # print(list(set(relabel_list)))
    print('最长句子为：', MaxLen)
    print(set(category_list))

    # 补全所有的句子
    total_data = new_total_data
    for num in total_data:
        difference = SentenceLength - len(total_data[num]['sentence'])
        total_data[num]['tokens_id'].extend([0] * difference)
        total_data[num]['tokens_class'].extend(['pad'] * difference)
        total_data[num]['tokens_class_num'].extend([class2num['pad']] * difference)
        total_data[num]['tokens_id'] = [str(x) for x in total_data[num]['tokens_id']]
        total_data[num]['tokens_class_num'] = [str(x) for x in total_data[num]['tokens_class_num']]

    # 将类型及编号进行存储
    with open(Class2NumFile, 'wb') as f:
        pickle.dump(class2num, f)

    for num in total_data:
        # rad = random.random()
        # if num > 3000 and rad < 0.02:
        #     if total_data[num]['sentence']:
        #         f_eval.write(total_data[num]['sentence'] + ',' +
        #                      ' '.join(total_data[num]['tokens_id']) + ',' +
        #                      ' '.join(total_data[num]['tokens_class']) + ',' +
        #                      ' '.join(total_data[num]['tokens_class_num']) + '\n'
        #                      )
        # else:
        if total_data[num]['sentence']:
            f_train.write(total_data[num]['sentence'] + ',' +
                          ' '.join(total_data[num]['tokens_id']) + ',' +
                          ' '.join(total_data[num]['tokens_class']) + ',' +
                          ' '.join(total_data[num]['tokens_class_num']) + '\n'
                          )


if __name__ == '__main__':
    print(len(open(VocabPath, 'r', encoding='utf-8').readlines()))
    parse_source_data()
    parse_new_data()
