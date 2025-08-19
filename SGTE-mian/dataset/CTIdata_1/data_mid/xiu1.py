import json
import re
# 确保 "triple_list" 中的第一个和第三个元素严格匹配 "text"（包括大小写）。如果它们不在 "text" 中，则删除该三元组，并过滤掉没有有效三元组的 JSON 对象。你可以运行它来清理你的数据！
# 如果对于那些字符多了的，我们只需要在三元组把这个删了
def is_whole_word_in_text(word, text):
    """检查word是否作为完整单词出现在text中（区分大小写）"""
    pattern = rf'\b{re.escape(word)}\b'
    return re.search(pattern, text) is not None


def filter_invalid_triples(data):
    filtered_data = []

    for item in data:
        text = item["text"]
        valid_triples = []

        for triple in item["triple_list"]:
            subject, relation, obj = triple

            # 确保 subject 和 obj 在 text 中是完整匹配
            if is_whole_word_in_text(subject, text) and is_whole_word_in_text(obj, text):
                valid_triples.append(triple)

        if valid_triples:
            filtered_data.append({"text": text, "triple_list": valid_triples})

    return filtered_data


# 读取 JSON 文件
json_path = "test.json"  # 替换为你的 JSON 文件路径
with open(json_path, "r", encoding="utf-8") as f:
    json_data = json.load(f)

# 过滤数据
filtered_json = filter_invalid_triples(json_data)

# 写入新的 JSON 文件
filtered_json_path = "test.json"
with open(filtered_json_path, "w", encoding="utf-8") as f:
    json.dump(filtered_json, f, ensure_ascii=False, indent=4)

print("数据过滤完成，已保存到 filtered_data.json")
