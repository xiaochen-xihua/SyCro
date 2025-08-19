import re
input_path = 'triples_unique.txt'
output_path = 'cleaned_triples.txt'


def clean_line(line):
    # 使用正则提取所有被双引号包裹的字段
    parts = re.findall(r'"(.*?)"', line)
    if len(parts) != 3:
        print("⚠️ 格式异常，跳过：", line.strip())
        return None
    # 替换空格和非法符号
    parts = [p.replace(" ", "_").replace(".", "").replace("’", "").replace("‘", "") for p in parts]
    return '\t'.join(parts)

with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        cleaned = clean_line(line)
        if cleaned:
            outfile.write(cleaned + '\n')

print("✅ 清洗完成，已保存为 cleaned_triples.txt")

