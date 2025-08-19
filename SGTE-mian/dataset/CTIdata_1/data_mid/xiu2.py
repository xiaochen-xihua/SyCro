import json

#这里生成模型可以接入的文件格式，new_xxx.txt(对应SPANRE模型数据格式)
def convert_json_format(data):
    converted_lines = []

    for item in data:
        sent_text = item["text"].strip("\b")  # 清理文本中的 \b
        relation_mentions = []

        for triple in item["triple_list"]:
            em1_text = triple[0].split()[-1]  # 取第一个实体的最后一个单词
            em2_text = triple[2].split()[-1]  # 取第二个实体的最后一个单词
            label = triple[1]  # 关系标签

            relation_mentions.append({
                "em1Text": em1_text,
                "em2Text": em2_text,
                "label": label
            })

        converted_line = json.dumps({
            "sentText": sent_text,
            "relationMentions": relation_mentions
        }, ensure_ascii=False)

        converted_lines.append(converted_line)

    return "\n".join(converted_lines)


# 读取原 JSON 数据
json1_path = "filtered_data.json"  # 你的 JSON 文件路径
with open(json1_path, "r", encoding="utf-8") as f:
    json1_data = json.load(f)

# 转换格式
converted_json2 = convert_json_format(json1_data)

# 写入新的 JSON 文件，每行一个 JSON 对象
json2_path = "new_valid.json"
with open(json2_path, "w", encoding="utf-8") as f:
    f.write(converted_json2)
