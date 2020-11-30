import time
import torch

cuda_condition = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda_condition else 'cpu')

VocabPath = 'checkpoint/pretrain/vocab_pretrain.txt'

# Debug开关
Debug = False

# 是否使用CRF
IsCrf = True

# 使用预训练模型开关
UsePretrain = True

# 任务模式
ModelClass = 'Bert'
AttentionMask = False

# ## NER训练通用参数开始 ## #
DropOut = 0.1
VocabSize = len(open(VocabPath, 'r', encoding='utf-8').readlines())
HiddenSize = 768
SentenceLength = 384
HiddenLayerNum = 12
IntermediateSize = 3072
AttentionHeadNum = 12
# ## NER训练通用参数结束 ## #

# ## NER训练调试参数开始 ## #
NEREpochs = 32
NERLearningRate = 1e-4
NerBatchSize = 1
# ## NER训练调试参数结束 ## #

# ## ner模型文件路径 ## #
NerSourcePath = 'data/source'
NerCorpusPath = 'data/train/ner_train.txt'
NerEvalPath = 'data/eval/ner_eval.txt'
NerTestPath = 'data/test'
Class2NumFile = 'data/train/c2n.pickle'
PretrainPath = 'checkpoint/pretrain/pytorch_bert_pretrain.bin'
NerFinetunePath = 'checkpoint/finetune/ner_trained_%s.model' % SentenceLength

# ## NER通用参数 ## #
NormalChar = 'ptzf'

# 参数名配对
local2target_emb = {
    'roberta_emb.token_embeddings.weight': 'bert.embeddings.word_embeddings.weight',
    # 'roberta_emd.type_embeddings.weight': 'bert.embeddings.token_type_embeddings.weight',
    # 'roberta_emd.position_embeddings.weight': 'bert.embeddings.position_embeddings.weight',
    'roberta_emb.emb_normalization.weight': 'bert.embeddings.LayerNorm.gamma',
    'roberta_emb.emb_normalization.bias': 'bert.embeddings.LayerNorm.beta'
}

local2target_transformer = {
    'transformer_blocks.%s.multi_attention.q_dense.weight': 'bert.encoder.layer.%s.attention.self.query.weight',
    'transformer_blocks.%s.multi_attention.q_dense.bias': 'bert.encoder.layer.%s.attention.self.query.bias',
    'transformer_blocks.%s.multi_attention.k_dense.weight': 'bert.encoder.layer.%s.attention.self.key.weight',
    'transformer_blocks.%s.multi_attention.k_dense.bias': 'bert.encoder.layer.%s.attention.self.key.bias',
    'transformer_blocks.%s.multi_attention.v_dense.weight': 'bert.encoder.layer.%s.attention.self.value.weight',
    'transformer_blocks.%s.multi_attention.v_dense.bias': 'bert.encoder.layer.%s.attention.self.value.bias',
    'transformer_blocks.%s.multi_attention.o_dense.weight': 'bert.encoder.layer.%s.attention.output.dense.weight',
    'transformer_blocks.%s.multi_attention.o_dense.bias': 'bert.encoder.layer.%s.attention.output.dense.bias',
    'transformer_blocks.%s.attention_layernorm.weight': 'bert.encoder.layer.%s.attention.output.LayerNorm.gamma',
    'transformer_blocks.%s.attention_layernorm.bias': 'bert.encoder.layer.%s.attention.output.LayerNorm.beta',
    'transformer_blocks.%s.feedforward.dense1.weight': 'bert.encoder.layer.%s.intermediate.dense.weight',
    'transformer_blocks.%s.feedforward.dense1.bias': 'bert.encoder.layer.%s.intermediate.dense.bias',
    'transformer_blocks.%s.feedforward.dense2.weight': 'bert.encoder.layer.%s.output.dense.weight',
    'transformer_blocks.%s.feedforward.dense2.bias': 'bert.encoder.layer.%s.output.dense.bias',
    'transformer_blocks.%s.feedforward_layernorm.weight': 'bert.encoder.layer.%s.output.LayerNorm.gamma',
    'transformer_blocks.%s.feedforward_layernorm.bias': 'bert.encoder.layer.%s.output.LayerNorm.beta',
}


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
