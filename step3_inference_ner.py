# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import pickle

from tqdm import tqdm
from extract_number import *
from pretrain_config import *
from roberta.common.tokenizers import Tokenizer
from step2_pretrain_ner import extract_output_entities


# def gen_test():
#     max_len = 0
#     result = {}
#     for file in os.listdir(NerTestPath):
#         if '.txt' in file:
#             file_num = file.split('.')[0]
#             result[file_num] = []
#             with open(os.path.join(NerTestPath, file), 'r', encoding='utf-8') as f:
#                 sentence = f.read().strip().replace(',', '，')
#                 if len(sentence) < SentenceLength:
#                     result[file_num].append(sentence)
#                     continue
#                 # 防止分割电影名
#                 safe = []
#                 sentence = list(sentence)
#                 for ci, c in enumerate(sentence):
#                     di = ci
#                     if c == '《':
#                         while di < len(sentence):
#                             if sentence[di] == '》':
#                                 safe = [ci, di]
#                                 break
#                             if sentence[di] == '《':
#                                 break
#                             di += 1
#                     else:
#                         if safe and safe[0] < ci < safe[1]:
#                             continue
#                         else:
#                             if c in ['，', '。', '…']:
#                                 sentence[ci] = '。'
#                 sentence = ''.join(sentence)
#                 ###################
#                 segments = sentence.split('。')
#                 seg_len = len(segments)
#                 tmp_str = segments[0]
#                 for i in range(1, seg_len):
#                     if tmp_str and len(tmp_str)+len(segments[i]) < SentenceLength - 1:
#                         tmp_str += segments[i] + '，'
#                     else:
#                         if tmp_str:
#                             result[file_num].append(tmp_str)
#                             if len(tmp_str) == 197:
#                                 print(tmp_str)
#                             tmp_str = ''
#                         else:
#                             tmp_str = segments[i]
#                 if tmp_str:
#                     result[file_num].append(tmp_str)
#                     if len(tmp_str) == 197:
#                         print(tmp_str)
#     for i in result:
#         for j in result[i]:
#             if len(j) > max_len:
#                 max_len = len(j)
#     print(max_len)
#     return result


def gen_test():
    max_len = 0
    result = {}
    for file in os.listdir(NerTestPath):
        if '.txt' in file:
            file_num = file.split('.')[0]
            result[file_num] = []
            with open(os.path.join(NerTestPath, file), 'r', encoding='utf-8') as f:
                sentence = f.read().strip().replace(',', '，')
                if len(sentence) < SentenceLength:
                    result[file_num].append(sentence)
                    continue
                segments = sentence.split('。')
                seg_len = len(segments)
                tmp_str = segments[0]
                for i in range(1, seg_len):
                    if tmp_str and len(tmp_str)+len(segments[i]) < SentenceLength - 1:
                        tmp_str += segments[i] + '。'
                    else:
                        if tmp_str:
                            result[file_num].append(tmp_str)
                            tmp_str = ''
                        else:
                            tmp_str = segments[i]
                if tmp_str:
                    result[file_num].append(tmp_str)
    for i in result:
        for j in result[i]:
            if len(j) > max_len:
                max_len = len(j)
            if len(j) > SentenceLength:
                print(i)
    print(max_len)
    return result


# def gen_entities(charclasses):
#     entities = {}
#     for i, cla in enumerate(charclasses):
#         if cla == NormalChar or cla == 'pad':
#             continue
#         if cla[0] == 'b':
#             j = i + 1
#             tmp = [cla]
#             current = cla[1:]
#             while j < len(charclasses):
#                 if charclasses[j][0]=='i' and charclasses[j][1:]==current:
#                     tmp.append(charclasses[j])
#                     j += 1
#                 elif charclasses[j][0]=='e' and charclasses[j][1:]==current:
#                     tmp.append(charclasses[j])
#                     entities[i] = tmp
#                     tmp = []
#                     break
#                 else:
#                     tmp = []
#                     break
#     return entities


class NerInference(object):
    def __init__(self):
        self.tokenizer = Tokenizer(VocabPath)
        with open(Class2NumFile, 'rb') as f:
            self.class_to_num = pickle.load(f)
        self.num_to_class = {}
        for k, v in self.class_to_num.items():
            self.num_to_class[v] = k
        try:
            self.model = torch.load(NerFinetunePath).to(device).eval()
        except:
            self.model = torch.load(NerFinetunePath, map_location='cpu').eval()
        print('加载模型完成！')

    def parse_inference_text(self, ori_line):
        ori_line = ori_line.strip().replace(' ', '')
        if len(list(ori_line)) > SentenceLength:
            print('文本过长！')
            return None, None

        input_tokens_id = []
        segment_ids = []
        for token in list(ori_line):
            id = self.tokenizer.token_to_id(token)
            input_tokens_id.append(id)

        for i in range(SentenceLength - len(input_tokens_id)):
            input_tokens_id.append(0)

        for x in input_tokens_id:
            if x:
                segment_ids.append(1)
            else:
                segment_ids.append(0)

        return input_tokens_id, segment_ids

    def inference_single(self, text):
        input_tokens_id, segment_ids = self.parse_inference_text(text)
        input_tokens_id = torch.tensor(input_tokens_id)
        segment_ids = torch.tensor(segment_ids)

        input_token = input_tokens_id.unsqueeze(0).to(device)
        segment_ids = torch.tensor(segment_ids).unsqueeze(0).to(device)
        input_token_list = input_token.tolist()
        input_len = len([x for x in input_token_list[0] if x])
        mlm_output = self.model(input_token, segment_ids)[:, :input_len, :]
        output_tensor = torch.nn.Softmax(dim=-1)(mlm_output)
        output_topk = torch.topk(output_tensor, 1).indices.squeeze(0).tolist()
        output2class = []
        for i, output in enumerate(output_topk):
            output = output[0]
            # output2class.append((text[i], self.num_to_class[output]))
            output2class.append(self.num_to_class[output])
        return output2class


if __name__ == '__main__':
    f = open('predict.csv', 'w', encoding='utf-8')
    f.write('ID,Category,Pos_b,Pos_e,Privacy\n')
    ner_infer = NerInference()
    test = gen_test()
    result = {}
    for num in tqdm(test):
        result[num] = []
        sentence = ''
        sentence_classes = []
        for segi, seg in enumerate(test[num]):
            if segi == 0:
                begin = 0
            else:
                begin = len(test[num][segi-1])
            re = ner_infer.inference_single(seg.lower().replace(',', '，'))
            qq = extract_qq(seg)
            if qq:
                for item in qq:
                    item[1] = begin + item[1]
                    item[2] = begin + item[2]
            for sub in qq:
                result[num].append([str(num)] + sub)

            mobile = extract_mobile(seg)
            if mobile:
                for item in mobile:
                    item[1] = begin + item[1]
                    item[2] = begin + item[2]
            for sub in mobile:
                result[num].append([str(num)] + sub)

            vx = extract_vx(seg)
            if vx:
                for item in vx:
                    item[1] = begin + item[1]
                    item[2] = begin + item[2]
            for sub in vx:
                result[num].append([str(num)] + sub)

            email = get_emailAddress(seg)
            if email:
                email[1] = begin + email[1]
                email[2] = begin + email[2]
                result[num].append([str(num)] + email)
            # print('\n')
            seg_class = ner_infer.inference_single(seg)
            sentence += seg
            sentence_classes.extend(seg_class)
        assert len(sentence) == len(sentence_classes)
        # 文本后处理
        # for i, charclass in enumerate(sentence_classes):
        #     if i > 0 and charclass[0] in ['b', 'i']:
        #         if i+1 < len(sentence_classes):
        #             if sentence_classes[i - 1][0] == 'e' \
        #                     and sentence_classes[i + 1][1:] != charclass[1:]\
        #                     and sentence_classes[i - 1][1:] == charclass[1:]:
        #                 sentence_classes[i - 1] = 'i' + sentence_classes[i - 1][1:]
        #                 sentence_classes[i] = 'e' + sentence_classes[i][1:]
        #         else:
        #             if sentence_classes[i - 1][0] == 'e' and sentence_classes[i - 1][1:] == charclass[1:]:
        #                 sentence_classes[i - 1] = 'i' + sentence_classes[i - 1][1:]
        #                 sentence_classes[i] = 'e' + sentence_classes[i][1:]
        entities = extract_output_entities(sentence_classes)
        # entities = gen_entities(sentence_classes)
        for pos in entities:
            if len(entities[pos]) == 1:
                continue
            result[num].append([str(num),
                                entities[pos][0][1:],
                                str(pos),
                                str(pos + len(entities[pos]) - 1),
                                sentence[pos:pos + len(entities[pos])]])

    for num in result:
        for item in result[num]:
            f.write(str(item[0])+','+str(item[1])+','+str(item[2])+','+str(item[3])+','+str(item[4])+'\n')
