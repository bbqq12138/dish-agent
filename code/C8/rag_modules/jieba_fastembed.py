import jieba
from fastembed import SparseTextEmbedding, SparseEmbedding

class JiebaFastEmbed:
    def __init__(self, model_name="Qdrant/bm25"):
        # 初始化底层的 fastembed 模型
        self._model = SparseTextEmbedding(model_name=model_name, cache_dir='/tmp/model_cache')
    
    def _preprocess(self, text: str) -> str:
        """这就是你以前那个 preprocess_func 的核心逻辑"""
        # 使用 jieba 精确模式分词，并用空格拼接
        return " ".join(jieba.cut(text))

    def embed_single(self, text: str) -> list[SparseEmbedding]:
        """处理单条文本（针对检索时的用户提问）"""
        processed_text = self._preprocess(text)
        # fastembed.embed 返回的是生成器，取第一个
        return list(self._model.embed([processed_text]))

    def embed_batch(self, texts: list[str]) -> list[SparseEmbedding]:
        """处理批量文本（针对批量菜谱入库）"""
        # 批量进行 jieba 分词
        processed_texts = [self._preprocess(t) for t in texts]
        # 让 fastembed 批量计算，利用底层 C++ 多线程优势
        return list(self._model.embed(processed_texts))