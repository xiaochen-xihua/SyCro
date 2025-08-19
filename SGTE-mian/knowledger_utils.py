import torch
import torch.nn as nn
import torch.nn.functional as F
from bert4keras.tokenizers import Tokenizer
# 这段代码实现了一个名为KnowledgeEmbedder的类，
# 主要用于加载、存储和操作不同类型的知识图谱嵌入（包括TransE、Node2Vec和GAT）。
# 该类的主要功能是提供对嵌入的查询，并能够根据输入的实体名称返回其对应的嵌入表示。下面是代码中各部分的详细分析：
class KnowledgeEmbedder(nn.Module):
    def __init__(self, transe_path, node2vec_path, gat_path, entity_triple_txt,tokenizer, device=None):
        super().__init__()  # ✅ 正确位置
        self.tokenizer = tokenizer
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # Load TransE
        transe_data = torch.load(transe_path,map_location=self.device)
        self.transe_entity2id = transe_data['entity2id']
        raw_transe_emb = transe_data['entity_embeddings']
        self.transe_entity_embeddings = raw_transe_emb.clone().detach().to(self.device)

        # Load Node2Vec
        node2vec_data = torch.load(node2vec_path,map_location=self.device)
        self.node2vec_nodes = node2vec_data['nodes']
        raw_node2vec_emb = node2vec_data['embeddings']
        self.node2vec_embeddings = raw_node2vec_emb.clone().detach().to(self.device)

        # Load GAT
        raw_gat = torch.load(gat_path, map_location='cpu')
        self.gat_embeddings = {}
        for k, v in raw_gat.items():
            self.gat_embeddings[k] = v.clone().detach().to(self.device)

        # Entity list
        self.entity_set = self.load_entity_list(entity_triple_txt)

        # Attention projection
        self.attn_proj = nn.Linear(3 * 128, 3, device=self.device)

    def load_entity_list(self, path):
        entity_set = set()
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # 直接整体作为实体，不拆分逗号
                    entity = line.strip().strip('"')
                    entity = self.normalize(entity)
                    entity_set.add(entity)
        return entity_set
# 从文本文件中加载实体集合（每行格式为 "entity", ...）。

    def is_entity(self, entity_name):
        return entity_name in self.entity_set

    def get_entity_embedding(self, entity_name):
        if not isinstance(entity_name, str):
            print(f"[警告] 非法实体名：{entity_name}")
            return None
        try:
            if entity_name not in self.transe_entity2id or \
               entity_name not in self.node2vec_nodes or \
               entity_name not in self.gat_embeddings:
                return None
            # return print("没有这个:",entity_name)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!用来检查知识三元组得构建有无错误

            # # 获取各个嵌入
            # transe_idx = self.transe_entity2id[entity_name]
            # transe_emb = self.transe_entity_embeddings[transe_idx].to(self.device)
            #
            # node2vec_idx = self.node2vec_nodes.index(entity_name)
            # node2vec_emb = self.node2vec_embeddings[node2vec_idx].to(self.device)
            #
            # gat_emb = self.gat_embeddings[entity_name].to(self.device)
            #
            # # 拼接用于计算注意力
            # attn_input = torch.cat([transe_emb, node2vec_emb, gat_emb], dim=-1)  # [384]
            # attn_weights = F.softmax(self.attn_proj(attn_input), dim=-1)  # [3]
            #
            # # 加权融合
            # fused = (
            #     attn_weights[0] * transe_emb +
            #     attn_weights[1] * node2vec_emb +
            #     attn_weights[2] * gat_emb
            # )  # [128]
            #
            # # 拼接残差
            # final_embedding = torch.cat([fused, transe_emb, node2vec_emb, gat_emb], dim=-1)  # [512]
            # return final_embedding

            transe_idx = self.transe_entity2id[entity_name]
            transe_emb = self.transe_entity_embeddings[transe_idx].to(self.device)
            # node2vec_idx = self.node2vec_nodes.index(entity_name)
            # node2vec_emb = self.node2vec_embeddings[node2vec_idx].to(self.device)

            #gat_emb = self.gat_embeddings[entity_name].to(self.device)
            #return gat_emb
            node2vec_idx = self.node2vec_nodes.index(entity_name)
            node2vec_emb = self.node2vec_embeddings[node2vec_idx].to(self.device)
            return node2vec_emb
        except Exception as e:
            print(f"Error retrieving embeddings for {entity_name}: {e}")
            return None
    # 这个代码是没有进行tokenizer更换的，因此只能匹配那些bert没有进行分词分开的实体词，比如###1就不行
    # def match_entity_spans(self, tokens: list[str]) -> list[tuple[int, int, str]]:
    #     spans = []
    #     token_len = len(tokens)
    #     entity_token_map = getattr(self, 'entity_token_map', None)
    #     if entity_token_map is None:
    #         entity_token_map = {
    #             entity: self.normalize(entity).split()
    #             for entity in self.entity_set
    #         }
    #
    #         # self.entity_token_map = entity_token_map  # 记忆下来，提高效率
    #     # 实体按分词长度从大到小排序，保证最长优先
    #     sorted_entities = sorted(entity_token_map.items(), key=lambda x: len(x[1]), reverse=True)
    #     pos = 0
    #     while pos < token_len:
    #         matched = False
    #         for entity, ent_tokens in sorted_entities:
    #             ent_len = len(ent_tokens)
    #             if ent_len == 0 or pos + ent_len > token_len:
    #                 continue
    #             window_tokens = tokens[pos:pos + ent_len]
    #             window_norm = [self.normalize(t) for t in window_tokens]
    #             if window_norm == ent_tokens:
    #                 # 详细打印
    #                 # print(
    #                 #     f"[MATCH] 实体: '{entity}' | tokens窗口: {window_tokens} | span索引: ({pos}, {pos + ent_len})")
    #                 # spans.append((pos, pos + ent_len, " ".join(window_tokens)))
    #                 spans.append((pos, pos + ent_len, entity))
    #                 pos += ent_len
    #                 matched = True
    #                 break  # 当前起点只取第一个（最长）匹配
    #         if not matched:
    #             pos += 1
    #     # 打印最终所有匹配结果
    #     # print(f"所有匹配spans: {spans}")
    #     return spans
    # def match_entity_spans(self, tokens: list[str]) -> list[tuple[int, int, str]]:
    #     spans = []
    #     token_len = len(tokens)
    #     entity_token_map = getattr(self, 'entity_token_map', None)
    #     if entity_token_map is None:
    #         entity_token_map = {
    #             entity: self.tokenizer.tokenize(entity)
    #             for entity in self.entity_set
    #         }
    #         self.entity_token_map = entity_token_map
    #
    #     sorted_entities = sorted(entity_token_map.items(), key=lambda x: len(x[1]), reverse=True)
    #
    #     pos = 0
    #     while pos < token_len:
    #         matched = False
    #         for entity, ent_tokens in sorted_entities:
    #             ent_len = len(ent_tokens)
    #             if ent_len == 0 or pos + ent_len > token_len:
    #                 continue
    #             window_tokens = tokens[pos:pos + ent_len]
    #             if window_tokens == ent_tokens:
    #                 spans.append((pos, pos + ent_len, entity))  # ✅ 用原始实体名 entity
    #                 pos += ent_len
    #                 matched = True
    #                 break
    #         if not matched:
    #             pos += 1
    #
    #     return spans

    def match_entity_spans(self, tokens: list[str]) -> list[tuple[int, int, str]]:
        """
        仅使用 tokenizer.tokenize(entity) 结果进行匹配，
        不做 normalize，不处理 ##，完全一致匹配。
        """
        spans = []
        token_len = len(tokens)

        # 实体名 -> 分词后的 token 列表
        entity_token_map = getattr(self, 'entity_token_map', None)
        if entity_token_map is None:
            def clean_tokens(toks):
                return [t for t in toks if t not in ['[CLS]', '[SEP]']]
            entity_token_map = {
                entity: clean_tokens(self.tokenizer.tokenize(entity))
                for entity in self.entity_set
            }
            self.entity_token_map = entity_token_map

        # 实体按长度从长到短排序
        sorted_entities = sorted(entity_token_map.items(), key=lambda x: len(x[1]), reverse=True)

        pos = 0
        while pos < token_len:
            matched = False
            for entity, ent_tokens in sorted_entities:
                ent_len = len(ent_tokens)
                if ent_len == 0 or pos + ent_len > token_len:
                    continue

                window_tokens = tokens[pos:pos + ent_len]

                # 直接严格比较 tokenizer 分词结果
                if window_tokens == ent_tokens:
                    spans.append((pos, pos + ent_len, entity))
                    # print(f"✅ 实体匹配成功：'{entity}' @ tokens[{pos}:{pos + ent_len}]")
                    pos += ent_len
                    matched = True
                    break
            if not matched:
                pos += 1

        if not spans:
            print("⚠️ 没有匹配到任何实体。tokens =", tokens)

        return spans
    @staticmethod
    def normalize(s: str) -> str:
        # 这里不小写，保留原样，去除BERT分词残留的##
        return s.replace("##", "").strip()

# ⚠️ 注意：我们统一对 token span 做了 normalize（小写、去掉BERT的“##”词缀）
