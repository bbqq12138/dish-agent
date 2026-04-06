from qdrant_client import AsyncQdrantClient, models
from langchain_huggingface import HuggingFaceEmbeddings
import asyncio
from rag_modules import DataPreparationModule, JiebaFastEmbed
from qdrant_client.models import PointStruct, SparseVector


qdrant_client = AsyncQdrantClient(url="http://localhost:6334", check_compatibility=False, prefer_grpc=True)
dense_model = HuggingFaceEmbeddings(
    model_name='BAAI/bge-large-zh-v1.5',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
sparse_model = JiebaFastEmbed(model_name='Qdrant/bm25')  # 重新封装的适合中文的稀疏向量模型
chunk_collection_name = "RecipeChunk"
data_module = DataPreparationModule("../../data/C8/cook")


async def store_chunks_in_qdrant():
    """将文档块存储到Qdrant中，使用稠密和稀疏向量"""
    # 加载文档块
    print("正在加载和分块文档...")
    data_module.load_documents()
    data_module.chunk_documents() 

    # 构建稠密、稀疏向量
    print("正在构建稠密和稀疏向量...")
    chunk_texts = [chunk.page_content for chunk in data_module.chunks]
    dense_vecs = await dense_model.aembed_documents(chunk_texts)
    sparse_vecs = sparse_model.embed_batch(chunk_texts)
    print(f"已构建 {len(dense_vecs)} 个稠密向量和 {len(sparse_vecs)} 个稀疏向量.")
    
    # 构建数据点对象
    print("正在构建Qdrant数据点对象...")
    points = []
    for dense_vec, sparse_vec, chunk in zip(dense_vecs, sparse_vecs, data_module.chunks):
        point = PointStruct(
            id = chunk.metadata['chunk_id'], 
            vector={
                'dense': dense_vec, 
                'sparse': SparseVector(
                    indices=sparse_vec.indices.tolist(), # toekn对应的索引
                    values=sparse_vec.values.tolist(),   # 索引对应的token的得分
                )
            }, 
            payload=chunk.metadata
        )
        points.append(point)
    print(f"已构建 {len(points)} 个数据点对象.")
        
    # 存储数据点到Qdrant
    print(f"正在将 {len(points)} 个数据点插入到 Qdrant 集合 '{chunk_collection_name}' 中...")
    await qdrant_client.upsert(
        collection_name=chunk_collection_name,
        points=points
    )
    print(f"已将 {len(points)} 个数据点插入到 Qdrant 集合 '{chunk_collection_name}' 中.")


async def main():
    if not await qdrant_client.collection_exists(collection_name=chunk_collection_name):
        print(f"集合 '{chunk_collection_name}' 不存在. 正在创建...")
        await qdrant_client.create_collection(
            collection_name=chunk_collection_name, 
            vectors_config={'dense': models.VectorParams(
                size=1024, 
                distance=models.Distance.COSINE, 
                on_disk=True,  # 稠密向量是否在内存，可以节省内存，牺牲时间
                datatype=models.Datatype.FLOAT32,  # 默认32,16节省内存
            )}, 
            sparse_vectors_config={'sparse': models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=True,  # 内存优化：稀疏索引放硬盘
                    datatype=models.Datatype.FLOAT32
                )
            )}, 
            hnsw_config=models.models.HnswConfigDiff(on_disk=True), 
            on_disk_payload=False  # 元数据直接存在内存中，元数据不大
        )

    await store_chunks_in_qdrant()

    
if __name__ == "__main__":
    asyncio.run(main())