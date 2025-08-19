import torch
import dgl
import dgl.nn as dglnn
import torch.nn as nn
import networkx as nx
import re
from collections import defaultdict

class DGLGATSubgraphEncoder(nn.Module):
    def __init__(self, in_feats=1, hidden_feats=64, out_feats=128, num_heads=4):
        super().__init__()
        self.gat_conv = dglnn.GATConv(in_feats, hidden_feats, num_heads)
        self.fc = nn.Linear(hidden_feats * num_heads, out_feats)

    def forward(self, g, features):
        h = self.gat_conv(g, features)
        h = h.flatten(1)
        out = self.fc(h)
        return out.mean(dim=0)  # aggregate subgraph representation

def load_triples(file_path):
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = re.findall(r'"(.*?)"', line.strip())
            if len(parts) == 3:
                triples.append(tuple(parts))
    return triples

def build_nx_graph(triples):
    G = nx.Graph()
    for h, r, t in triples:
        G.add_edge(h, t)
    return G

def convert_nx_to_dgl(subgraph):
    mapping = {node: i for i, node in enumerate(subgraph.nodes())}
    dgl_graph = dgl.graph([(mapping[u], mapping[v]) for u, v in subgraph.edges()])
    dgl_graph = dgl.to_simple(dgl_graph)
    return dgl_graph, mapping

def save_dgl_gat_subgraph_embeddings(G, entities, out_path='coarse_grained_subgraph_dgl.pt'):
    model = DGLGATSubgraphEncoder()
    model.eval()
    embeddings = {}
    for node in entities:
        if node not in G:
            continue
        sub_nodes = list(nx.single_source_shortest_path_length(G, node, cutoff=1).keys())
        subgraph = G.subgraph(sub_nodes).copy()
        if subgraph.number_of_nodes() == 0:
            continue
        dgl_g, mapping = convert_nx_to_dgl(subgraph)
        x = torch.ones((dgl_g.num_nodes(), 1))  # Use constant input feature
        with torch.no_grad():
            emb = model(dgl_g, x)
        embeddings[node] = emb
    torch.save(embeddings, out_path)

def main():
    triples = load_triples("../triples_unique.txt")
    G = build_nx_graph(triples)
    entities = list(set([h for h, _, _ in triples] + [t for _, _, t in triples]))
    save_dgl_gat_subgraph_embeddings(G, entities)

if __name__ == "__main__":
    main()