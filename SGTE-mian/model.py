from transformers.models.bert import BertModel, BertPreTrainedModel,BertTokenizer,BertTokenizerFast
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import (
    BertSelfAttention,
    BertAttention,
    BertIntermediate,
    BertOutput
)
from ikan.FourierKAN import FourierKAN
from ikan.TaylorKAN import TaylorKAN
from knowledger_utils import KnowledgeEmbedder
from TAGCN import TypeAttentiveGCN,DependencyProcessor
from util import *

class DecoderLayer(nn.Module):    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.KAN_proj = FourierKAN(
            layers_hidden=[config.hidden_size,config.hidden_size],
        )
        self.gate = nn.Parameter(torch.tensor(0.5))  # 初始化为 0.5（可调）
        self.layernorm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states,  #B num_generated_triples H(维度)
        encoder_hidden_states,
        encoder_attention_mask
    ):
        self_attention_outputs = self.attention(hidden_states)         attention_output = self_attention_outputs[0] # hidden_states.shape

        attention_output = self.KAN_proj(attention_output)
        kan_output = self.KAN_proj(attention_output)
        attention_output = attention_output + kan_output  # 残差连接
        gate = torch.sigmoid(self.gate)  # shape: scalar
        attention_output = attention_output + gate * kan_output
        attention_output = self.layernorm(attention_output + gate * kan_output)

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights


        # 扩展编码器注意力掩码
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :] # B 1 1 H
        else:
            raise ValueError(
                "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                    encoder_hidden_shape, encoder_attention_mask.shape
                )
            )
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0 #1 1 0 0 -> 0 0 -1000 -1000

        # 交叉注意力机制
        cross_attention_outputs = self.crossattention(
            hidden_states=attention_output, encoder_hidden_states=encoder_hidden_states,  encoder_attention_mask=encoder_extended_attention_mask
        ) # 将 encoder 的输出作为“记忆”，进一步引入上下文信息（常见于 encoder-decoder 架构）
        attention_output = cross_attention_outputs[0] #B m H
        outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        #kan_output = self.KAN_proj(attention_output)
        # 残差连接
        #attention_output = attention_output + kan_output #用不了一点，因为内存不够大

        # 中间层和输出层
        intermediate_output = self.intermediate(attention_output)

        # kan_output = self.KAN_proj(intermediate_output)
        # # 残差连接
        # intermediate_output = intermediate_output + kan_output 用不了一点，因为内存不够大

        layer_output = self.output(intermediate_output, attention_output) #B m H
        # 和标准 Transformer 一样，做两层线性层 + 残差连接。
        outputs = (layer_output,) + outputs
        return outputs

class SGTE(BertPreTrainedModel):
    def __init__(self, config, tokenizer):
        super(GRTE, self).__init__(config)
        self.tokenizer = tokenizer
        self.tokenizer1 = BertTokenizerFast.from_pretrained("bert-base-cased")
        self.ta_gcn = TypeAttentiveGCN(input_dim=config.hidden_size,
                                     hidden_dim=config.hidden_size) 
        self.dep_processor = DependencyProcessor() 
        self.bert=BertModel(config=config)
        if config.fix_bert_embeddings:
            self.bert.embeddings.word_embeddings.weight.requires_grad = False
            self.bert.embeddings.position_embeddings.weight.requires_grad = False
            self.bert.embeddings.token_type_embeddings.weight.requires_grad = False

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kb = KnowledgeEmbedder(
            transe_path='konw/knowtrip/transe_embedding.pt',
            node2vec_path='konw/knowtrip/mid_grained_node2vec.pt',
            gat_path='konw/knowtrip/coarse_grained_subgraph_dgl.pt',
            entity_triple_txt='konw/knowtrip/triple.txt',
            tokenizer=self.tokenizer,
            device=device  
        )
        # 融合层：拼接 BERT 输出 + 知识向量，映射回 BERT 原维度
        self.knowledge_attn = nn.Linear(config.hidden_size, config.hidden_size)
        # self.knowledge_key_proj = nn.Linear(512, config.hidden_size)  # 投影知识向量
        self.knowledge_key_proj = nn.Linear(128, config.hidden_size)  # 投影知识向量
        self.knowledge_gate = nn.Sequential(
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        # 统一 new_embed 和 enhanced_embed 分布，通过线性映射
        self.pre_fuse_proj1 = nn.Linear(768, 768)
        self.pre_fuse_proj2 = nn.Linear(768, 768)

        self.fusion_gate_layer = nn.Linear(768 * 2, 1)

        # 融合构造的代码部分
        self.fusion_layer1 = nn.Linear(4 * config.hidden_size, 2 * config.hidden_size)
        self.fusion_layer2 = nn.Linear(2 * config.hidden_size, config.hidden_size)



        self.Lr_e1=nn.Linear(config.hidden_size,config.hidden_size)
        self.Lr_e2=nn.Linear(config.hidden_size,config.hidden_size)

        self.elu=nn.ELU()
        self.Cr = nn.Linear(config.hidden_size, config.num_p*config.num_label)

        self.Lr_e1_rev=nn.Linear(config.num_p*config.num_label,config.hidden_size)
        self.Lr_e2_rev=nn.Linear(config.num_p*config.num_label,config.hidden_size)

        self.rounds=config.rounds

        self.e_layer=DecoderLayer(config)

        torch.nn.init.orthogonal_(self.Lr_e1.weight, gain=1)
        torch.nn.init.orthogonal_(self.Lr_e2.weight, gain=1)
        torch.nn.init.orthogonal_(self.Cr.weight, gain=1)
        torch.nn.init.orthogonal_(self.Lr_e1_rev.weight, gain=1)
        torch.nn.init.orthogonal_(self.Lr_e2_rev.weight, gain=1)

    def forward(self, token_ids, mask_token_ids):
        tokens_str_batch = [
            [self.tokenizer.id_to_token(int(i)) for i in ids]
            for ids in token_ids
        ]
        embed = self.get_embed(token_ids, mask_token_ids,tokens_str_batch)
        #embed:BLH
        L=embed.shape[1]

        e1 = self.Lr_e1(embed) # BLL H
        e2 = self.Lr_e2(embed)

        for i in range(self.rounds):
            h = self.elu(e1.unsqueeze(2).repeat(1, 1, L, 1) * e2.unsqueeze(1).repeat(1, L, 1, 1))  # BLL 2H
            B, L = h.shape[0], h.shape[1]

            table_logist = self.Cr(h)  # BLL RM

            if i!=self.rounds-1:

                table_e1 = table_logist.max(dim=2).values
                table_e2 = table_logist.max(dim=1).values
                e1_ = self.Lr_e1_rev(table_e1)
                e2_ = self.Lr_e2_rev(table_e2)

                e1=e1+self.e_layer(e1_,embed,mask_token_ids)[0]
                e2=e2+self.e_layer(e2_,embed,mask_token_ids)[0]

        return table_logist.reshape([B,L,L,self.config.num_p,self.config.num_label])

    def get_embed(self, token_ids, mask_token_ids, tokens_str_batch):
        bert_out = self.bert(input_ids=token_ids.long(), attention_mask=mask_token_ids.long())
        embed = self.dropout(bert_out[0])  # (B, L, 768)
        new_embed = embed.clone() # 用于被修改，不破坏原始 BERT 输出
        new_embed1 = embed.clone()  # 用于被修改，不破坏原始 BERT 输出
        for b in range(embed.shape[0]): # 逐个处理 batch 中的句子（对每一句进行知识融合）
            tokens_str = tokens_str_batch[b]
            spans = self.kb.match_entity_spans(tokens_str)
            for start, end, ent_str in spans:
                kb_emb = self.kb.get_entity_embedding(ent_str)
                if kb_emb is None:
                    continue
                kb_emb = kb_emb.to(embed.device)  # (384,)
                projected_kb = self.knowledge_key_proj(kb_emb)  
                span_tokens = embed[b, start:end]  
                projected_span = self.knowledge_attn(span_tokens)  
                attn_scores = torch.matmul(projected_span, projected_kb)  # (span_len,)
                attn_weights = torch.softmax(attn_scores, dim=0)  # (span_len,)
                attn_output = torch.sum(attn_weights.unsqueeze(-1) * span_tokens, dim=0)  # (768,)
                # gated fusion
                gate_input = torch.cat([attn_output, projected_kb], dim=-1) 
                gate = self.knowledge_gate(gate_input)
                fused = gate * attn_output + (1 - gate) * projected_kb  # (768,)
                new_embed[b, start] = fused 
        batch_enhanced = []
        for i in range(new_embed1.size(0)):
            sentence = self.decode_tokens(token_ids[i])  
            adj, dep_types = self.dep_processor.process_sentence(sentence)
            # === 对齐维度 ===
            h = new_embed1[i]  # [L, H]
            sent_len = mask_token_ids[i].sum().item()
            h = h[:sent_len]  # 有效 token 的表示 [L', H]

            # 对齐 adj 和 dep_types 到 h
            n = h.size(0)
            if adj.size(0) != n:
                if adj.size(0) > n:
                    adj = adj[:n, :n]
                    dep_types = dep_types[:n, :n]
                else:
                    pad_n = n - adj.size(0)
                    adj = F.pad(adj, (0, pad_n, 0, pad_n))  # 上下左右 pad
                    dep_types = F.pad(dep_types, (0, pad_n, 0, pad_n))

            adj = adj.to(h.device)
            dep_types = dep_types.to(h.device)

            # === 表示增强 ===
            h_enhanced = self.ta_gcn(h, adj, dep_types)  # [n, H]

            # === 固定长度填充（到100） ===
            final_embed = torch.zeros(100, h_enhanced.size(-1)).to(h.device)
            final_embed[:n] = h_enhanced
            batch_enhanced.append(final_embed)
        enhanced_embed = torch.stack(batch_enhanced)

        # 统一 new_embed 和 enhanced_embed 分布，因为这两个的差异比较大
        new_embed_proj = self.pre_fuse_proj1(new_embed)  # [B, L, H]
        enhanced_embed_proj = self.pre_fuse_proj2(enhanced_embed)  # [B, L, H]
        # h_kb: [B, L, H] - 知识增强后
        # h_dep: [B, L, H] - 句法依存增强后
        assert new_embed.shape == enhanced_embed.shape  # 必须形状相同
        fusion_input = torch.cat([new_embed_proj, enhanced_embed_proj], dim=-1)
        # 门控：计算融合权重 → [B, L, 1]
        gate = torch.sigmoid(self.fusion_gate_layer(fusion_input))
        # 加权融合 → [B, L, H]
        fused = gate * new_embed + (1 - gate) * enhanced_embed
        # [B, L, 2H] → [B, L, H]
        return fused
    def decode_tokens(self, token_ids):
        return recover_text_from_tokens(token_ids, self.tokenizer1)