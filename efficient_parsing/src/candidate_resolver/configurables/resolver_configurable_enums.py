from enum import Enum


class EmbeddingType(Enum):
    SEQUENCE = "SEQ"
    BERT_POOLED = "BERT_POOL"
    SEQUENCE_POSITIONAL = "SEQ_POS"
    MAX_POOLED = "MAX_POOL"
    AVG_POOLED = "AVG_POOL"
    MAX_POOLED_POSITIONAL = "MAX_POOL_POS"
    AVG_POOLED_POSITIONAL = "AVG_POOL_POS"
    SUMMED_QUADRATIC_KERNEL_SEQUENCE = "SUM_QUAD_SEQ"
    SUMMED_QUADRATIC_KERNEL_BERT_POOLED = "SUM_QUAD_BERT_POOLED"
    SUMMED_QUADRATIC_KERNEL_SEQUENCE_POSITIONAL = "SUM_QUAD_SEQ_POS"


class BertKerasLayerType(Enum):
    BERT_SMALL = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/2"
    BERT_MEDIUM = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
    BERT_LARGE = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/4"
    ELECTRA_SMALL = "https://tfhub.dev/google/electra_small/2"
    ELECTRA_MEDIUM = "https://tfhub.dev/google/electra_base/2"
    ELECTRA_LARGE = "https://tfhub.dev/google/electra_large/2"


class MetricLearnerType(Enum):
    NONE = 0
    NCA = 1
    LMNN = 2
