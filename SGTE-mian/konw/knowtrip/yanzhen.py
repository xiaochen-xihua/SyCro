import torch

def check_transe(file='transe_embedding.pt'):
    print("🔎 Checking TransE (细粒度) embeddings...")
    data = torch.load(file)
    ent = data['entity_embeddings']
    rel = data['relation_embeddings']
    print(f"Entities: {ent.shape}, Relations: {rel.shape}")
    print("Sample entity:", list(data['entity2id'].keys())[:3])
    return data


def check_node2vec(file='mid_grained_node2vec.pt'):
    print("🔎 Checking Node2Vec (中粒度) embeddings...")
    data = torch.load(file)
    embeddings = data['embeddings']
    nodes = data['nodes']

    print(f"Entities: {len(nodes)}")
    print("Sample nodes:", nodes[:3])
    print("Embedding shape:", embeddings.shape)
    return {'embeddings': embeddings, 'nodes': nodes}


def check_gat(file='coarse_grained_subgraph_dgl.pt'):
    print("🔎 Checking GAT Subgraph (粗粒度) embeddings...")
    data = torch.load(file)
    keys = list(data.keys())[:3]
    print(f"Entities: {len(data)}")
    print("Sample:", keys)
    print("Embedding shape:", data[keys[0]].shape)
    return data

def main():
    transe = check_transe()
    node2vec = check_node2vec()
    gat = check_gat()

    entity = "A.S. Roma"
    print(f"\n🎯 Entity check: {entity}")
    if entity in transe['entity2id']:
        idx = transe['entity2id'][entity]
        print("TransE:", transe['entity_embeddings'][idx][:5])
    if entity in node2vec['nodes']:
        idx = node2vec['nodes'].index(entity)
        print("Node2Vec:", node2vec['embeddings'][idx][:5])
    if entity in gat:
        print("GAT:", gat[entity][:5])

if __name__ == "__main__":
    main()
