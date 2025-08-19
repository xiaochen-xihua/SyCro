import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

class KnowCTIEmbedder(nn.Module):
    def __init__(self, embed_dim=768, gat_hidden=256, gat_heads=4):
        super(KnowCTIEmbedder, self).__init__()
        self.embed_dim = embed_dim
        self.gat = GATConv(embed_dim, gat_hidden, heads=gat_heads, concat=True)
        self.output_layer = nn.Linear(gat_hidden * gat_heads + embed_dim, embed_dim)  # concat fusion

    def forward(self, text_embeddings, triples_batch):
        """
        text_embeddings: [B, L, D] = [6, 100, 768]
        triples_batch: List[List[Triples]] for each sample in batch, each triple: (head_idx, rel, tail_idx)
        """

        B, L, D = text_embeddings.size()
        output = []

        for i in range(B):
            embeddings = text_embeddings[i]  # [L, D]
            triples = triples_batch[i]       # list of (h, r, t)

            if not triples:
                output.append(embeddings)
                continue

            # Prepare graph: build edge_index for GAT
            edges = []
            for h, r, t in triples:
                edges.append((h, t))  # assuming directional h → t
                edges.append((t, h))  # add reverse to make undirected

            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # shape [2, num_edges]

            # Build Data object for GAT
            x = embeddings  # node features [L, D]
            data = Data(x=x, edge_index=edge_index)

            # Run GAT
            x_gat = self.gat(data.x, data.edge_index)  # [L, gat_hidden * heads]

            # Concatenate GAT output with original embedding
            x_fused = torch.cat([embeddings, x_gat], dim=-1)  # [L, D + GAT_output_dim]
            x_out = self.output_layer(x_fused)                # [L, D]
            output.append(x_out)

        return torch.stack(output, dim=0)  # [B, L, D]


