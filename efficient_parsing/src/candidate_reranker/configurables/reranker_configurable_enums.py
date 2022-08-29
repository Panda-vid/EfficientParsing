from enum import Enum


class LambdaEmbedderAttached(Enum):
    NO = False
    YES = True


class TableEmbedderPoolingType(Enum):
    AVG = "AVG"
    MAX = "MAX"


class TableEmbeddingContent(Enum):
    COLUMN_ONLY = True
    CONTENT_INCLUDED = False
