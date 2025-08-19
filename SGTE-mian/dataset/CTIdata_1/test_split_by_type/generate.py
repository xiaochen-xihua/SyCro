import json

def is_normal_triple(triples):
    """每对实体（头、尾）只出现一次，没有多标签、重叠"""
    entity_pairs = set()
    for triple in triples:
        pair = (triple[0], triple[2])
        entity_pairs.add(pair)
    # 如果每个pair只有一个对应关系，并且三元组数=pair数，则是normal
    return len(entity_pairs) == len(triples)

def is_multi_label(triples):
    """同一对实体有多个关系"""
    pair_relations = {}
    for triple in triples:
        pair = (triple[0], triple[2])
        pair_relations.setdefault(pair, set()).add(triple[1])
    # 有pair有多个relation
    return any(len(rels) > 1 for rels in pair_relations.values())

def is_over_lapping(triples):
    """实体出现在多个三元组（作为头或尾）"""
    entity_count = {}
    for triple in triples:
        for ent in (triple[0], triple[2]):
            entity_count[ent] = entity_count.get(ent, 0) + 1
    # 有实体出现超过一次
    return any(count > 1 for count in entity_count.values())

def convert_and_save_by_type(input_file, out_normal, out_multi, out_overlap):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(out_normal, "w", encoding="utf-8") as f_normal, \
         open(out_multi, "w", encoding="utf-8") as f_multi, \
         open(out_overlap, "w", encoding="utf-8") as f_overlap:
        for item in data:
            sent = item["text"]
            triples = item.get("triple_list", [])
            relation_mentions = [
                {
                    "em1Text": triple[0],
                    "em2Text": triple[2],
                    "label": triple[1]
                }
                for triple in triples
            ]
            out = {
                "sentText": sent,
                "relationMentions": relation_mentions
            }
            if is_normal_triple(triples):
                f_normal.write(json.dumps(out, ensure_ascii=False) + "\n")
            if is_multi_label(triples):
                f_multi.write(json.dumps(out, ensure_ascii=False) + "\n")
            if is_over_lapping(triples):
                f_overlap.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    # 文件名可按需更改
    input_file = '../test_triples.json'
    output_normal = 'test_triples_normal.json'
    output_multi = 'test_triples_epo.json'
    output_overlap = 'test_triples_seo.json'
    convert_and_save_by_type(input_file, output_normal, output_multi, output_overlap)
