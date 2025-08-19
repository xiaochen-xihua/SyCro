import torch
# 检查 TransE 文件
transe_data = torch.load("transe_embedding_2.pt")
print(transe_data['entity_embeddings'].device)  # 应该是 cpu 或 cuda:X，不能是 meta

# 检查 Node2Vec 文件
node2vec_data = torch.load("mid_grained_node2vec_2.pt")
print(node2vec_data['embeddings'].device)

# 检查 GAT 文件
gat_data = torch.load("coarse_grained_subgraph_dgl_2.pt")
print(next(iter(gat_data.values())).device)