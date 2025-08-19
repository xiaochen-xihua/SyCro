#转化下数据库的存储形式
import re
from difflib import SequenceMatcher
from pathlib import Path


def extract_first_quoted(text):
    """提取第一个双引号内的内容"""
    matches = re.findall(r'"([^"]*)"', text)
    return matches[0] if matches else None


def group_similar_strings(strings, threshold=0.6):
    """基于相似度分组"""
    groups = []
    for s in sorted(strings, key=len, reverse=True):
        matched = False
        for group in groups:
            if SequenceMatcher(None, s, group[0]).ratio() >= threshold:
                group.append(s)
                matched = True
                break
        if not matched:
            groups.append([s])
    return groups


def process_file(input_path, output_path):
    # 读取输入文件
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 提取所有第一个引号内容并去重
    strings = list(set(
        extract_first_quoted(line)
        for line in lines if extract_first_quoted(line)
    ))

    # 分组并排序
    groups = group_similar_strings(strings)
    sorted_groups = sorted(groups, key=lambda g: -max(len(s) for s in g))
    result = [
        f'"{s}"'
        for group in sorted_groups
        for s in sorted(group, key=len, reverse=True)
    ]

    # 写入输出文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(result))


if __name__ == "__main__":
    input_file = "mid.txt"  # 你的输入文件路径
    output_file = "processed_triples.txt"  # 输出文件路径
    process_file(input_file, output_file)
    print(f"处理完成！结果已保存到 {output_file}")