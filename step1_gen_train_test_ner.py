import os
import pickle
import random

from tqdm import tqdm
from pretrain_config import *
from roberta.common.tokenizers import Tokenizer


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
    f_eval = open(NerEvalPath, 'w', encoding='utf-8')
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
                if 'e' + category not in class2num:
                    class2num['e' + category] = len(class2num)
                total_data[int(file_num)]['tokens_class'][end] = 'e' + category
                total_data[int(file_num)]['tokens_class_num'][end] = class2num['e' + category]
            if end - begin == 1:
                if 'b' + category not in class2num:
                    class2num['b' + category] = len(class2num)
                if 'e' + category not in class2num:
                    class2num['e' + category] = len(class2num)
                # 更新tokens_class
                total_data[int(file_num)]['tokens_class'][begin] = 'b' + category
                total_data[int(file_num)]['tokens_class'][end] = 'e' + category
                # 更新tokens_class_num
                total_data[int(file_num)]['tokens_class_num'][begin] = class2num['b' + category]
                total_data[int(file_num)]['tokens_class_num'][end] = class2num['e' + category]
            if end - begin > 1:
                if 'b' + category not in class2num:
                    class2num['b' + category] = len(class2num)
                if 'i' + category not in class2num:
                    class2num['i' + category] = len(class2num)
                if 'e' + category not in class2num:
                    class2num['e' + category] = len(class2num)
                total_data[int(file_num)]['tokens_class'][begin] = 'b' + category
                total_data[int(file_num)]['tokens_class'][begin+1:end] = ['i' + category] * (end - begin - 1)
                total_data[int(file_num)]['tokens_class'][end] = 'e' + category
                total_data[int(file_num)]['tokens_class_num'][begin] = class2num['b' + category]
                total_data[int(file_num)]['tokens_class_num'][begin+1:end] = [class2num['i' + category]] * (end - begin - 1)
                total_data[int(file_num)]['tokens_class_num'][end] = class2num['e' + category]

    # 将长句进行分割
    new_total_data = {}
    tmp_docker = ['', [], [], []]
    for num in total_data:
        if len(total_data[num]['sentence']) <= 128:
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
                    if tc[i][0] in ['i', 'e'] or 0 < len(tmp_docker[0]) < 10:
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
        rad = random.random()
        if num > 3000 and rad < 0.02:
            if total_data[num]['sentence']:
                f_eval.write(total_data[num]['sentence'] + ',' +
                             ' '.join(total_data[num]['tokens_id']) + ',' +
                             ' '.join(total_data[num]['tokens_class']) + ',' +
                             ' '.join(total_data[num]['tokens_class_num']) + '\n'
                             )
        else:
            if total_data[num]['sentence']:
                f_train.write(total_data[num]['sentence'] + ',' +
                              ' '.join(total_data[num]['tokens_id']) + ',' +
                              ' '.join(total_data[num]['tokens_class']) + ',' +
                              ' '.join(total_data[num]['tokens_class_num']) + '\n'
                              )


if __name__ == '__main__':
    print(len(open(VocabPath, 'r', encoding='utf-8').readlines()))
    parse_source_data()
