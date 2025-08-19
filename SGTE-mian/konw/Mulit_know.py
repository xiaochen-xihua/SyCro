import networkx as nx
from node2vec import Node2Vec
import json
import re


# 读取知识库txt文件并解析三元组
def parse_triples(file_path):
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 用正则提取三个字段
            parts = re.findall(r'"(.*?)"', line)
            if len(parts) == 3:
                h, r, t = parts
                triples.append((h.strip(), r.strip(), t.strip()))
    return triples


# 构建知识图谱图
def build_kg_graph(triples):
    G = nx.MultiDiGraph()
    for h, r, t in triples:
        G.add_edge(h, t, key=r)
    return G


# 使用 Node2Vec 生成中粒度路径嵌入
def generate_node2vec_embeddings(G, dimensions=64):
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=10, num_walks=100, workers=1)
    model = node2vec.fit(window=5, min_count=1)
    return model


# 示例：提取某个实体的子图
def extract_subgraph(G, entity, hops=1):
    nodes = nx.single_source_shortest_path_length(G, entity, cutoff=hops).keys()
    return G.subgraph(nodes)


# 示例：使用子图特征（节点度数作为粗粒度嵌入）
def subgraph_embedding_simple(subgraph):
    node_degrees = [d for _, d in subgraph.degree()]
    return sum(node_degrees) / len(node_degrees) if node_degrees else 0


# 主流程
if __name__ == "__main__":
    file_path = 'knowledge_triples.txt'

    print("📥 Parsing triples...")
    triples = parse_triples(file_path)

    print("🔧 Building knowledge graph...")
    G = build_kg_graph(triples)

    print("🚶 Generating Node2Vec embeddings...")
    n2v_model = generate_node2vec_embeddings(G)

    # 示例：查询一个实体（如 "Jens Härtel"）
    entity = "Jens Härtel"

    print(f"🔎 Getting embeddings for entity: {entity}")
    if entity in n2v_model.wv:
        embedding = n2v_model.wv[entity]
        print("✅ Path-based (Node2Vec) embedding:", embedding[:10], "...")
    else:
        print("⚠️ Entity not in embedding model.")

    print(f"🌲 Extracting subgraph around: {entity}")
    subG = extract_subgraph(G, entity)
    sub_emb = subgraph_embedding_simple(subG)
    print("✅ Subgraph-level embedding (avg degree):", sub_emb)
