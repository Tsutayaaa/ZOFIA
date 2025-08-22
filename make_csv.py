import os
import csv
import re
from pathlib import Path
from collections import defaultdict

# === 手动指定多个 output 文件夹 ===
BATCH_DIRS = [
    "./batch_output(R3)",
    "./batch_output(R4)",
    # 添加更多路径
]

# === 初始化事件统计表 ===
event_table = defaultdict(dict)  # {sample: {batch: count}}

# === 遍历每个指定的 batch 文件夹 ===
for batch_path in BATCH_DIRS:
    batch_folder = Path(batch_path)
    if not batch_folder.is_dir():
        print(f"⚠️ 跳过无效路径: {batch_path}")
        continue
    batch_name = batch_folder.name
    print(f"📂 正在处理: {batch_name}")

    for sample_folder in sorted(batch_folder.glob("*")):
        if not sample_folder.is_dir():
            continue
        sample_name = sample_folder.name
        summary_file = sample_folder / f"{sample_name}_summary.txt"
        if summary_file.exists():
            with open(summary_file, "r") as f:
                first_line = f.readline().strip()
                match = re.search(r"Total detected events:\s*(\d+)", first_line)
                if match:
                    count = int(match.group(1))
                    event_table[sample_name][batch_name] = count
                else:
                    print(f"⚠️ 无法解析事件数: {summary_file}")
        else:
            print(f"⚠️ 未找到文件: {summary_file}")

# === 获取所有样本名和批次名 ===
all_samples = sorted(event_table.keys())
all_batches = sorted({b for counts in event_table.values() for b in counts})

# === 写入 CSV ===
output_csv = "event_counts_summary_1.csv"
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Sample"] + all_batches)
    for sample in all_samples:
        row = [sample] + [event_table[sample].get(batch, "") for batch in all_batches]
        writer.writerow(row)

print(f"✅ 汇总完成: {output_csv}")