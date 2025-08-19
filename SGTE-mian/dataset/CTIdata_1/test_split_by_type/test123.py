import json

def count_text_in_json(input_file):
    try:
        # 打开并读取 JSON 文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)  # 解析 JSON 数据

        # 统计 "text" 键的数量
        text_count = sum(1 for item in data if 'text' in item)

        print(f"Total number of 'text' entries: {text_count}")
        return text_count

    except FileNotFoundError:
        print(f"Error: The file {input_file} does not exist.")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode the JSON file {input_file}.")

# 示例调用
input_file = 'test_triples_seo.json'  # 替换为你的文件路径
count_text_in_json(input_file)
