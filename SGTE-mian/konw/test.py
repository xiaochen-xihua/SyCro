import torch
#
# # 加载模型文件
# data = torch.load("transe_embeddings.pt", map_location="cpu")
#
# # 打印所有 key
# print("Keys in transe_embeddings.pt:")
# print(data.keys())
import torch


# data = torch.load("transe_embeddings.pt", map_location="cpu")
#
# # 获取实体名称映射（如果键不同，请修改为实际键名）
# entity2id = data["ent2id"]
#
# # 搜索包含 "peter" 或 "stöger" 的实体
# for name in entity2id:
#     if "peter" in name.lower() or "stöger" in name.lower():
#         print(name)
# 加载模型文件
data = torch.load("knowtrip/transe_embedding.pt", map_location="cpu")

# 获取实体嵌入和映射
entity_embeddings = data["entity_embeddings"]       # Tensor: [num_entities, 768]
entity2id = data["entity2id"]                       # dict: entity_name -> index

# 你要查看的实体名
entity_name = "Peter Stöger"  # 举例：你可以换成任何实体名

# 检查是否存在
if entity_name in entity2id:
    idx = entity2id[entity_name]
    embedding_vector = entity_embeddings[idx]  # Tensor: [768]
    print(f"Entity: {entity_name}")
    print(f"Embedding (shape={embedding_vector.shape}):\n{embedding_vector}")
else:
    print(f"实体 `{entity_name}` 不在 entity2id 映射中，请检查拼写或格式。")
