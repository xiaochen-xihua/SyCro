import json

# 读取原始数据（列表格式）
with open('../test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('output.json', 'w', encoding='utf-8') as fout:
    for item in data:
        out = {
            "sentText": item["text"],
            "relationMentions": [
                {
                    "em1Text": triple[0],
                    "em2Text": triple[2],
                    "label": triple[1]
                } for triple in item.get("triple_list", [])
            ]
        }
        fout.write(json.dumps(out, ensure_ascii=False) + '\n')

print("转换完成！每行是一个新的JSON对象。")
