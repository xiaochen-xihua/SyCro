import networkx as nx
from node2vec import Node2Vec
import torch
import re

def load_triples(file_path):
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = re.findall(r'"(.*?)"', line.strip())
            if len(parts) == 3:
                triples.append(tuple(parts))
    return triples

def build_graph(triples):
    G = nx.Graph()
    for h, r, t in triples:
        G.add_edge(h, t)
    return G

def save_node2vec_embeddings(G, path='mid_grained_node2vec.pt', dim=128):
    node2vec = Node2Vec(G, dimensions=dim, walk_length=10, num_walks=100, workers=1)
    model = node2vec.fit(window=5, min_count=1)
    embeddings = {node: model.wv[node] for node in model.wv.index_to_key}
    torch.save(embeddings, path)

def main():
    triples = load_triples("triples_unique.txt")
    G = build_graph(triples)
    save_node2vec_embeddings(G)

if __name__ == "__main__":
    main()
