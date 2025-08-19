import torch
import torch.nn as nn
import torch.optim as optim
import re
from tqdm import tqdm

# 读取三元组
def load_triples(file_path):
    triples = []
    entities = set()
    relations = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = re.findall(r'"(.*?)"', line.strip())
            if len(parts) == 3:
                h, r, t = parts
                triples.append((h, r, t))
                entities.add(h)
                entities.add(t)
                relations.add(r)
    return triples, sorted(list(entities)), sorted(list(relations))

# 建立字典
def build_vocab(items):
    return {item: idx for idx, item in enumerate(items)}

# TransE模型
class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, emb_dim=128):
        super().__init__()
        self.ent_embeddings = nn.Embedding(num_entities, emb_dim)
        self.rel_embeddings = nn.Embedding(num_relations, emb_dim)
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

    def forward(self, head, rel, tail):
        score = self.ent_embeddings(head) + self.rel_embeddings(rel) - self.ent_embeddings(tail)
        return torch.norm(score, p=2, dim=1)

# 负采样
def negative_sampling(triples_idx, num_entities):
    neg_triples = []
    for h, r, t in triples_idx:
        if torch.rand(1).item() < 0.5:
            h_ = torch.randint(0, num_entities, (1,)).item()
            neg_triples.append((h_, r, t))
        else:
            t_ = torch.randint(0, num_entities, (1,)).item()
            neg_triples.append((h, r, t_))
    return neg_triples

# 训练
def train_transe(triples, entity2id, relation2id, emb_dim=128, epochs=200, lr=0.01):
    num_entities = len(entity2id)
    num_relations = len(relation2id)
    model = TransE(num_entities, num_relations, emb_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    margin = 1.0
    loss_fn = nn.MarginRankingLoss(margin=margin)

    triples_idx = [(entity2id[h], relation2id[r], entity2id[t]) for h, r, t in triples]
    triples_idx = torch.tensor(triples_idx)

    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        neg_triples = negative_sampling(triples_idx, num_entities)
        neg_triples = torch.tensor(neg_triples)

        pos_h, pos_r, pos_t = triples_idx[:,0], triples_idx[:,1], triples_idx[:,2]
        neg_h, neg_r, neg_t = neg_triples[:,0], neg_triples[:,1], neg_triples[:,2]

        pos_scores = model(pos_h, pos_r, pos_t)
        neg_scores = model(neg_h, neg_r, neg_t)

        target = torch.ones(pos_scores.size())
        loss = loss_fn(pos_scores, neg_scores, target)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}, loss={loss.item():.4f}")

    return model

# 保存嵌入
def save_embeddings(model, entity2id, relation2id, path='transe_embedding.pt'):
    torch.save({
        'entity_embeddings': model.ent_embeddings.weight.data.cpu(),
        'relation_embeddings': model.rel_embeddings.weight.data.cpu(),
        'entity2id': entity2id,
        'relation2id': relation2id
    }, path)

# 查询函数
def load_embeddings(path='transe_embedding_2.pt'):
    data = torch.load(path)
    return data

def get_entity_embedding(name, data):
    idx = data['entity2id'].get(name)
    if idx is None:
        return None
    return data['entity_embeddings'][idx]

def get_relation_embedding(name, data):
    idx = data['relation2id'].get(name)
    if idx is None:
        return None
    return data['relation_embeddings'][idx]

# 主流程
def main():
    triples, entities, relations = load_triples('../triples_unique.txt')
    entity2id = build_vocab(entities)
    relation2id = build_vocab(relations)
    model = train_transe(triples, entity2id, relation2id, emb_dim=128, epochs=200, lr=0.01)
    save_embeddings(model, entity2id, relation2id)

    # 查询示例
    data = load_embeddings()
    emb = get_entity_embedding("South Africa", data)
    print("South Africa embedding shape:", emb.shape)
    print("South Africa embedding (前5维):", emb[:5])

if __name__ == "__main__":
    main()
